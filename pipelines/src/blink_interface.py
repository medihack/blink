from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
import nibabel as nb
import numpy as np
import scipy.stats as stats
from scipy import ndimage
import csv
import os
import re
from collections import OrderedDict


def calc_center_of_region(region_id, atlas):
    # clone atlas data
    atlas_data = np.copy(atlas)

    # mask region id
    atlas_data[atlas_data != region_id] = 0
    atlas_data[atlas_data != 0] = 1

    # calculate center of mass
    center = ndimage.measurements.center_of_mass(atlas_data)
    x = int(np.round(center[0]))
    y = int(np.round(center[1]))
    z = int(np.round(center[2]))

    return (x, y, z)


def parse_defined_regions(fname, atlas=None):
    regions = OrderedDict()

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
                if atlas == None:
                    raise Exception("No coordinates for region %i, but also no atlas provided to calculate them." % region_id)
                coords = calc_center_of_region(region_id, atlas)

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


def write_mapped_regions(mapped_regions):
    fname = os.path.join(os.getcwd(), "mapped_regions.txt")
    with open(fname, 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        for region_id, region in mapped_regions.iteritems():
            writer.writerow(region)
    return fname


def write_matrix(matrix, fname):
    fname = os.path.join(os.getcwd(), fname)
    with open(fname, "w") as f:
        for row in matrix:
            row = map(str, row)
            line = " ".join(row)
            f.write(line + "\n")
    return fname


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
        regions = list()
        with open(fname) as f:
            freader = csv.reader(f, delimiter='\t')
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
            fname = "region_mask_%i.nii.gz" % (int(region_id))
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
        desc="zero values on diagonal in output matrices"
    )


class FunctionalConnectivityOutputSpec(TraitedSpec):
    matrix = File(
        desc="the connectivity matrix (calculated using the pearson correlation)"
    )
    normalized_matrix = File(
        desc="the normalized connectivity matrix (Fisher Z transformed)"
    )
    p_values = File(
        desc="the matrix of p-values for the pearson correlated connectivity matrix"
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

        defined_regions = parse_defined_regions(
            self.inputs.defined_regions, atlas_data)

        results = self.gen_fconn(defined_regions, fmri_data, atlas_data)

        matrix = results["corr_matrix"]
        self.matrix = write_matrix(matrix, "matrix.txt")

        p_values = results["p_values"]
        self.p_values = write_matrix(p_values, "p_values.txt")

        normalized_matrix = results["normalized_matrix"]
        self.normalized_matrix = write_matrix(normalized_matrix, "normalized_matrix.txt")

        self.mapped_regions = write_mapped_regions(defined_regions)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["matrix"] = self.matrix
        outputs["p_values"] = self.p_values
        outputs["normalized_matrix"] = self.normalized_matrix
        outputs["mapped_regions"] = self.mapped_regions
        return outputs

    def gen_fconn(self, defined_regions, fmri_data, atlas_data):
        # some validation checkings
        if len(fmri_data.shape) != 4:
            raise Exception("fMRI data must have four dimensions (spatial + temporal)")

        if len(atlas_data.shape) != 3:
            raise Exception("Atlas data must have three dimensions (spatial)")

        if fmri_data.shape[:3] != atlas_data.shape:
            raise Exception("fMRI and atlas data must have same spatial dimensions")

        region_ids = defined_regions.keys()

        # calculate mean values for all atlas region_ids over time
        timepoints = fmri_data.shape[3]
        means = np.zeros([len(region_ids), timepoints], "float32")

        for i in xrange(timepoints):
            fmri_vol = fmri_data[:, :, :, i]

            for j, region_id in enumerate(region_ids):
                mask = np.copy(atlas_data)
                mask[atlas_data != region_id] = 0

                if np.all(mask == 0):
                    raise Exception("Defined region id (%i) not in atlas." % (region_id))

                masked = np.copy(fmri_vol)
                masked[mask == 0] = 0

                mean = masked.mean()
                means[j, i] = mean

        # calculate pearson correlation
        mat = np.ones([len(region_ids), len(region_ids)])
        pv = np.zeros([len(region_ids), len(region_ids)])

        for i in xrange(len(region_ids)):
            for j in xrange(len(region_ids)):
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
            "corr_matrix": mat,
            "p_values": pv,
            "normalized_matrix": zmat
        }


class StructuralConnectivityInputSpec(BaseInterfaceInputSpec):
    targets = traits.List(
        exists=True,
        desc="seeds to target output from probtrackx",
        mandatory=True
    )
    logs = traits.List(File(exists=True),
        exists=True,
        desc="logs from probtrackx",
        mandatory=True
    )
    defined_regions = File(
        exists=True,
        desc="a text file that defines the regions in the atlas",
        mandatory=True
    )
    atlas = File(
        exists=True,
        desc='an atlas for calculating the regions center',
        mandatory=False
    )
    zero_diagonal = traits.Bool(
        usedefault=True,
        default_value=True,
        desc="zero values on diagonal in output matrices"
    )
    # not used currently as we normalize by the waystotal
    # of a region connected to each self
    # (this should be the same as voxels_of_region * n_samples)
    #n_samples = traits.Int(
        #exists=True,
        #desc='the number of samples used in probtrackx',
        #mandatory=True
    #)


class StructuralConnectivityOutputSpec(TraitedSpec):
    matrix = File(
        desc="the matrix containing the total number of ways (per two regions)"
    )
    normalized_matrix = File(
        desc="the normalized connectivity matrix"
    )
    mapped_regions = File(
        desc="the regions represented in the matrix (as defined in the atlas)"
    )

class StructuralConnectivity(BaseInterface):
    """
    Creates a structural connectvity matrix from the results of probtrackx.
    Also maps provided region definitions to the connectivity matrix
    (see FunctionalConnectivity class).
    """
    input_spec = StructuralConnectivityInputSpec
    output_spec = StructuralConnectivityOutputSpec

    def _run_interface(self, runtime):
        targets = self.inputs.targets
        seed_ids = self.parse_seed_ids(self.inputs.logs)

        atlas_fname = self.inputs.atlas
        if atlas_fname:
            atlas_img = nb.load(atlas_fname)
            atlas_data = atlas_img.get_data()

        defined_regions = parse_defined_regions(
            self.inputs.defined_regions, atlas_data)

        results = self.gen_sconn(defined_regions, seed_ids, targets)

        matrix = results["waystotal_matrix"]
        self.matrix = write_matrix(matrix, "matrix.txt")

        normalized_matrix = results["normalized_matrix"]
        self.normalized_matrix = write_matrix(normalized_matrix, "normalized_matrix.txt")

        self.mapped_regions = write_mapped_regions(defined_regions)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["matrix"] = self.matrix
        outputs["normalized_matrix"] = self.normalized_matrix
        outputs["mapped_regions"] = self.mapped_regions
        return outputs

    def parse_seed_ids(self, logs):
        seed_ids = list()
        for log in logs:
            with open(log) as f:
                l = f.readline()
                r = r'.*--seed\S+region_mask_(\d+)\.nii.*'
                m = re.match(r, l)
                seed_id = int(m.group(1))
                assert seed_id
                seed_ids.append(seed_id)
        return seed_ids

    def gen_sconn(self, defined_regions, seed_ids, targets):
        assert len(seed_ids) == len(targets)
        d = dict()
        region_ids = defined_regions.keys()
        ways_mat = np.zeros([len(region_ids), len(region_ids)])
        norm_mat = np.zeros([len(region_ids), len(region_ids)])

        for idx, seed_id in enumerate(seed_ids):
            if not seed_id in region_ids:
                continue

            seed_targets = targets[idx]
            for seed_target in seed_targets:
                r = r'.*seeds_to_region_mask_(\d+)\.nii.*'
                m = re.match(r, seed_target)
                target_id = int(m.group(1))
                assert target_id

                if not target_id in region_ids:
                    continue

                (size, ways) = self.analyze_target(seed_target)

                ds = d.get(seed_id, dict())
                dt = ds.get(target_id, dict())
                dt = {'size': size, 'ways': ways}
                ds[target_id] = dt
                d[seed_id] = ds

        # calculate matrices
        for i in xrange(len(region_ids)):
            for j in xrange(len(region_ids)):
                r1 = region_ids[i]
                r2 = region_ids[j]

                # waytotals matrix
                # (seed -> target) + (target -> seed)
                w1 = d[r1][r2]['ways']
                w2 = d[r2][r1]['ways']
                waystotal = w1 + w2
                ways_mat[i, j] = ways_mat[j, i] = waystotal

                # normalized matrix
                n1 = d[r1][r1]['ways']
                n2 = d[r2][r2]['ways']
                norm = (((w1 / n1) + (w2 / n2)) / 2)
                norm_mat[i, j] = norm_mat[j, i] = norm

        # postprocess matrices
        if self.inputs.zero_diagonal:
            np.fill_diagonal(ways_mat, 0)
            np.fill_diagonal(norm_mat, 0)

        return {
            "waystotal_matrix": ways_mat,
            "normalized_matrix": norm_mat
        }

    def analyze_target(self, seed_target):
        target_img = nb.load(seed_target)
        target_data = target_img.get_data()
        size = len(target_data[target_data!=0])
        ways = np.sum(target_data)
        return (size, ways)
