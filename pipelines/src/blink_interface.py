from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
import nibabel as nb
import numpy as np
import scipy.stats as stats
from scipy import ndimage
import csv
import os

class AtlasMergerInputSpec(BaseInterfaceInputSpec):
    atlas1 = File(exists=True, desc="1st input atlas")
    atlas2 = File(exists=True, desc="2nd input atlas")
    regions1 = File(exists=True, desc="1st input region description")
    regions2 = File(exists=True, desc="2nd input region description")


class AtlasMergerOutputSpec(TraitedSpec):
    merged_atlas = File(desc="merged atlas")
    merged_regions = File(desc="merged regions of both atlases")


class AtlasMerger(BaseInterface):
    """
    Merges two atlases into a combined one. Output is new nifti combined atlas file and
    new list of combined regions with new numbering, as well as new nift of old atlas with new numbering.
    The first input atlas is the priority atlas, so regions defined in this atlas
    will win in a conflicting situation over regions in the second atlas.
    Zero, "0", must be nothing or air, in the atlas and region definition file.
    """
    input_spec = AtlasMergerInputSpec
    output_spec = AtlasMergerOutputSpec

    def _run_interface(self, runtime):
        atlas1_fname = self.inputs.atlas1
        atlas2_fname = self.inputs.atlas2
        atlas1_img = nb.load(atlas1_fname)
        atlas2_img = nb.load(atlas2_fname)
        atlas1_data = atlas1_img.get_data()
        atlas2_data = atlas2_img.get_data()
        regions1_fname = self.inputs.regions1
        regions2_fname = self.inputs.regions2

        if atlas1_data.size != atlas2_data.size:
            raise Exception("Atlases must have the same resolution - please resample!")

        regions1 = self.parse_regions(regions1_fname)
        regions2 = self.parse_regions(regions2_fname)

        (new_atlas1, new_regions1, startpos) = self.redefine_regions(atlas1_data, regions1)
        (new_atlas2, new_regions2, x) = self.redefine_regions(atlas2_data, regions2, startpos)

        merged_atlas = self.merge_atlases(new_atlas1, new_atlas2)
        self.merged_atlas = self.write_atlas(merged_atlas,"merged_atlas.nii.gz",atlas1_img)
        #self.write_atlas(new_atlas1,"newatlas1", atlas1_img)
        #self.write_atlas(new_atlas2,"newatlas2",atlas2_img)

        merged_regions = self.merge_regions(merged_atlas, new_regions1, new_regions2)
        self.merged_regions = self.write_regions(merged_regions, "merged_regions.csv")

        return runtime

    def parse_regions(self, fname):
        with open(fname) as f:
            freader = csv.reader(f, delimiter='\t')
            regions = list()
            for row in freader:
                regions.append(row)
        return regions

    def redefine_regions(self, atlas, regions, startpos=1):
        new_atlas = np.zeros(atlas.shape, 'int')
        new_regions = list()
        for region in regions:
            new_atlas[atlas==int(region[0])] = startpos
            new_region = list(region)
            new_region[0] = startpos
            new_regions.append(new_region)
            startpos += 1
        return (new_atlas, new_regions, startpos)

    def merge_atlases(self, new_atlas1, new_atlas2):
        shape = new_atlas1.shape
        merged_atlas = np.zeros(shape, 'int')
        for slice_index, slice_value in enumerate(merged_atlas):
            for row_index, row_value in enumerate(slice_value):
                for voxel_index, voxel_value in enumerate(row_value):
                    voxel1 = new_atlas1[slice_index, row_index, voxel_index]
                    voxel2 = new_atlas2[slice_index, row_index, voxel_index]
                    if voxel1 > 0:
                        merged_atlas[slice_index, row_index, voxel_index] = voxel1
                    elif voxel2 > 0:
                        merged_atlas[slice_index, row_index, voxel_index] = voxel2
        return merged_atlas

    def write_atlas(self, atlas, atlas_fname, oldatlas_img):
        fname = os.path.join(os.getcwd(), atlas_fname)
        nb.Nifti1Image(
            atlas,
            oldatlas_img.get_affine(),
            oldatlas_img.get_header()
        ).to_filename(fname)
        return fname

    def merge_regions(self, merged_atlas, new_regions1, new_regions2):
        unique = np.unique(merged_atlas)
        region_ids = np.delete(unique, np.where(unique == 0))
        merged_regions = list()
        for region_id in region_ids:
            r = filter(lambda r: int(r[0]) == region_id, new_regions1)
            if not r:
                r = filter(lambda r: int(r[0]) == region_id, new_regions2)
            if r:
                merged_regions.append(r[0])
        return merged_regions

    def write_regions(self, merged_regions, merged_regions_fname):
        fname = os.path.join(os.getcwd(), merged_regions_fname)
        with open(fname, 'w') as f:
            fwriter = csv.writer(f, delimiter='\t')
            for row in merged_regions:
                fwriter.writerow(row)
        return fname

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["merged_atlas"] = self.merged_atlas
        outputs["merged_regions"] = self.merged_regions
        return outputs


class AtlasSplitterInputSpec(BaseInterfaceInputSpec):
    atlas = File(exists=True, desc="an atlas volume to split by region")


class AtlasSplitterOuputSpec(TraitedSpec):
    masks = traits.List(File(), desc="a mask for each region in the provided atlas")
    mappings = traits.Dict(
        traits.Either(traits.CInt(), traits.CFloat()),
        File(),
        desc="mapping of region id to mask file")

class AtlasSplitter(BaseInterface):
    """
    Splits a atlas and creates a mask for each region in the atlas.
    Also provides a dictionary to associate each region id with
    the appropriate mask file.

    Example
    -------

    >>> import nipype.interfaces.blink as blink
    >>> asplit = blink.AtlasSplitter()
    >>> asplit.inputs.atlas = 'aal.nii'
    >>> asplit.run()
    """
    input_spec = AtlasSplitterInputSpec
    output_spec = AtlasSplitterOuputSpec

    def _run_interface(self, runtime):
        atlas_fname = self.inputs.atlas
        atlas_img = nb.load(atlas_fname)

        self.split_atlas(atlas_img)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["masks"] = self.masks
        outputs["mappings"] = self.mappings
        return outputs

    def split_atlas(self, atlas_img):
        atlas_data = atlas_img.get_data()

        # calculate number of unique regions in atlas
        # (assumes that 0 is air)
        unique = np.unique(atlas_data.flatten())
        regions = np.delete(unique, np.where(unique == 0))
        regions = np.sort(regions)

        # calculate masks
        masks = dict()
        for region_id in regions:
            mask = np.copy(atlas_data)
            mask[atlas_data != region_id] = 0
            mask[mask != 0] = 1
            masks[region_id] = mask

        # save masks to disc and create mappings
        mappings = dict()
        for region_id, mask in masks.items():
            fname = "region_mask_%s.nii.gz" % (region_id)
            fname = os.path.join(os.getcwd(), fname)

            nb.Nifti1Image(
                mask,
                atlas_img.get_affine(),
                atlas_img.get_header()
            ).to_filename(fname)

            mappings[region_id] = fname

        self.mappings = mappings
        self.masks = mappings.values()


class FunctionalConnectivityInputSpec(BaseInterfaceInputSpec):
    fmri = File(
        exists=True,
        desc='the fMRI volume',
        mandatory=True
    )
    atlas = File(
        exists=True,
        desc='an atlas that defines the regions',
        mandatory=True
    )
    defined_regions = File(
        exists=True,
        desc="a text file that defines the regions in the atlas",
    )
    absolute = traits.Bool(
        usedefault=True,
        default_value=True,
        desc="absolute values in correlation and normalized matrix"
    )
    zero_diagonal = traits.Bool(
        usedefault=True,
        default_value=True,
        desc="zero values on diagonal in correlation and normalized matrix"
    )


class FunctionalConnectivityOutputSpec(TraitedSpec):
    matrix = File(
        desc="the connectivity matrix (calculated using the pearson correlation)"
    )
    p_values = File(
        desc="the matrix of p-values for the pearson correlated connectivity matrix"
    )
    normalized_matrix = File(
        desc="the normalized connectivity matrix (Fisher Z transformed)"
    )
    mapped_regions = File(
        desc="the regions represented in the matrix (as defined in the atlas)"
    )


class FunctionalConnectivity(BaseInterface):
    """
    Creates a (functional) connectivity matrix from a fMRI volume using an atlas volume (for defining regions).
    The fMRI volume and the atlas must have the same resolution.
    Each distinct value (!= 0) in the atlas volume is treated as region (0 is air).
    Also maps provided region definitions to the connectivity matrix.
    A definitions file is a CSV file (one region per line) with the following data (\t is tab):
        id\tlabel\tfull_name\tx\ty\tz
    For example a line to map AAL region 68 (left precuneus) would look like this:
        67, PCUN.L, Precuneus, -7, -56, -48
    If no coordinates (x, y, z) for this region are provided then those will be
    automatically calculated by using the center of mass of the region.

    Example
    -------

    >>> import nipype.interfaces.blink as blink
    >>> fconn = blink.FunctionalConnectivity()
    >>> fconn.inputs.fmri = 'fmri.nii'
    >>> fconn.inputs.atlas = 'aal.nii'
    >>> fconn.run()
    """
    input_spec = FunctionalConnectivityInputSpec
    output_spec = FunctionalConnectivityOutputSpec

    def _run_interface(self, runtime):
        fmri_fname = self.inputs.fmri
        fmri_img = nb.load(fmri_fname)
        fmri_data = fmri_img.get_data()

        atlas_fname = self.inputs.atlas
        atlas_img = nb.load(atlas_fname)
        atlas_data = atlas_img.get_data()
        self.atlas_data = atlas_data

        defined_regions_fname = self.inputs.defined_regions
        defined_regions = self.parse_defined_regions(defined_regions_fname)

        results = self.gen_fconn(fmri_data, atlas_data)

        matrix = results["matrix"]
        self.matrix = self.write_matrix(matrix, "matrix.txt")

        p_values = results["p_values"]
        self.p_values = self.write_matrix(p_values, "p_values.txt")

        normalized_matrix = results["normalized_matrix"]
        self.normalized_matrix = self.write_matrix(normalized_matrix, "normalized_matrix.txt")

        region_ids = results["region_ids"]
        mapped_regions = self.map_regions(defined_regions, region_ids)
        self.mapped_regions = self.write_mapped_regions(mapped_regions)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["matrix"] = self.matrix
        outputs["p_values"] = self.p_values
        outputs["normalized_matrix"] = self.normalized_matrix
        outputs["mapped_regions"] = self.mapped_regions
        return outputs

    def calc_center_of_region(self, region_id):
        # clone atlas data
        atlas_data = np.copy(self.atlas_data)

        # mask region id
        atlas_data[atlas_data != region_id] = 0
        atlas_data[atlas_data != 0] = 1

        # calculate center of mass
        center = ndimage.measurements.center_of_mass(atlas_data)
        x = int(np.round(center[0]))
        y = int(np.round(center[1]))
        z = int(np.round(center[2]))

        return (x, y, z)

    def parse_defined_regions(self, fname):
        regions = dict()

        with open(fname) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                region_id = int(row[0].strip())

                try:
                    x = int(row[3].strip())
                    y = int(row[4].strip())
                    z = int(row[5].strip())
                    coords = (x, y, z)
                except IndexError:
                    coords = self.calc_center_of_region(region_id)

                region = (
                    row[1].strip(),  # label
                    row[2].strip(),  # full_name
                    coords[0],  # x
                    coords[1],  # y
                    coords[2],  # z
                )

                if region_id in regions:
                    raise Exception("Duplicate region id in region definitions file: %i" % region_id)
                regions[region_id] = region

        return regions

    def gen_fconn(self, fmri_data, atlas_data):
        # some validation checkings
        if len(fmri_data.shape) != 4:
            raise Exception("fMRI data must have four dimensions (spatial + temporal)")

        if len(atlas_data.shape) != 3:
            raise Exception("Atlas data must have three dimensions (spatial)")

        if fmri_data.shape[:3] != atlas_data.shape:
            raise Exception("fMRI and atlas data must have same spatial dimensions")

        # calculate number of unique regions in atlas
        # (assumes that 0 is air)
        unique = np.unique(atlas_data.flatten())
        regions = np.delete(unique, np.where(unique == 0))
        regions = np.sort(regions)

        # calculate mean values for all atlas regions over time
        timepoints = fmri_data.shape[3]
        means = np.zeros([regions.size, timepoints], "float32")

        for i in xrange(timepoints):
            fmri_vol = fmri_data[:, :, :, i]

            for j, val in enumerate(regions):
                mask = np.copy(atlas_data)
                mask[atlas_data != val] = 0

                masked = np.copy(fmri_vol)
                masked[mask == 0] = 0

                mean = masked.mean()
                means[j, i] = mean

        # calculate pearson correlation
        mat = np.ones([regions.size, regions.size])
        pv = np.zeros([regions.size, regions.size])

        for i in xrange(regions.size):
            for j in xrange(regions.size):
                x = means[i, :]
                y = means[j, :]
                corr = stats.pearsonr(x, y)
                mat[i, j] = mat[j, i] = corr[0]
                pv[i, j] = pv[j, i] = corr[1]

        assert (mat == mat.transpose()).all()
        assert mat.shape[0] == mat.shape[1]
        assert (mat.diagonal() == 1).all()

        # normalize matrix
        zmat = np.copy(mat)
        np.fill_diagonal(zmat, 0)
        zmat = np.arctanh(zmat) # Fisher Z transformation
        weights = zmat[np.triu_indices_from(zmat, 1)]
        mean = weights.mean()
        zmat = zmat - mean
        np.fill_diagonal(zmat, 1)

        assert (zmat == zmat.transpose()).all()
        assert zmat.shape[0] == zmat.shape[1]
        assert (zmat.diagonal() == 1).all()

        # postprocess matrices
        if self.inputs.absolute:
            mat = np.absolute(mat)
            zmat = np.absolute(zmat)

        if self.inputs.zero_diagonal:
            np.fill_diagonal(mat, 0)
            np.fill_diagonal(zmat, 0)

        return {
            "matrix": mat,
            "p_values": pv,
            "normalized_matrix": zmat,
            "region_ids": regions
        }

    def write_matrix(self, matrix, fname):
        fname = os.path.join(os.getcwd(), fname)
        with open(fname, "w") as f:
            for row in matrix:
                row = map(str, row)
                line = " ".join(row)
                f.write(line + "\n")
        return fname

    def map_regions(self, defined_regions, region_ids):
        mapped_regions = list()

        for region_id in region_ids:
            region_id = int(region_id)
            region = defined_regions[region_id]
            if not region:
                raise Exception("Missing definition for region id: %i" % region_id)
            mapped_regions.append(region)

        return mapped_regions

    def write_mapped_regions(self, mapped_regions):
        fname = os.path.join(os.getcwd(), "mapped_regions.txt")
        with open(fname, 'w') as f:
            writer = csv.writer(f, delimiter="\t")
            for region in mapped_regions:
                writer.writerow(region)
        return fname
