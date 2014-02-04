from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
import nibabel as nb
import numpy as np
import scipy.stats as stats
from scipy import ndimage
import csv
import os


class AtlasSplitterInputSpec(BaseInterfaceInputSpec):
    atlas = File(exists=True, desc="an atlas volume to split by region")


class AtlasSplitterOuputSpec(TraitedSpec):
    masks = File(desc="a mask for each region in the provided atlas")
    mappings = traits.Dict(desc="mapping of region id to mask file")


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
            fname = "region_mask_%s" % (region_id)
            fname = os.path.join(os.getcwd(), fname)

            nb.Nifti1Image(
                mask,
                atlas_img.get_affine(),
                atlas_img.get_header()
            ).to_filename(fname)

            mappings[region_id] = fname

        self.mappings = mappings
        self.masks = mappings.values()


class RegionsMapperInputSpec(BaseInterfaceInputSpec):
    regions = traits.Array(exists=True, desc="the region ids to map")
    definitions = File(exists=True, desc="the definition of the regions in tab-delimited format")
    atlas = File(desc="an atlas volume for finding the center of a region")


class RegionsMapperOutputSpec(TraitedSpec):
    mapped_regions = traits.Array(desc="the mapped regions")


class RegionsMapper(BaseInterface):
    """
    Maps given region identifiers to the specified region informations.
    A definitions file is a CSV file (one region per line) with the following data:
        id, label, full_name, x, y, z
    For example a line to map AAL region 68 (left precuneus) would look like this:
        67, PCUN.L, Precuneus, -7, -56, -48
    If no coordinates (x, y, z) for this region are provided then those will be
    automatically calculated by using the center of mass of the region.

    Example
    -------

    >>> import nipype.interfaces.blink as blink
    >>> rmap = blink.RegionsMapper()
    >>> rmap.inputs.regions = [3, 4, 5, 9]
    >>> rmap.inputs.definitions = 'regions.csv'
    >>> rmap.inputs.atlas = 'aal.nii'
    >>> rmap.run()
    """
    input_spec = RegionsMapperInputSpec
    output_spec = RegionsMapperOutputSpec

    def _run_interface(self, runtime):
        atlas = self.inputs.atlas
        if atlas:
            self.atlas_data = self.load_atlas(atlas)

        definitions = self.load_definitions()

        self.mapped_regions = self.map_regions(definitions)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["mapped_regions"] = self.mapped_regions
        return outputs

    def load_atlas(self, atlas_fname):
        atlas_img = nb.load(atlas_fname)
        atlas_data = atlas_img.get_data()
        return atlas_data

    def load_definitions(self):
        definitions = dict()
        def_fname = self.inputs.definitions
        with open(def_fname, "rb") as def_file:
            reader = csv.reader(def_file, delimiter="\t")
            for row in reader:
                region_id = row[0]

                try:
                    x = int(row[3])
                    y = int(row[4])
                    z = int(row[5])
                    coords = (x, y, z)
                except IndexError:
                    coords = self.calc_center_of_region(region_id)

                d = dict(
                    label=row[1],
                    full_name=row[2],
                    x=coords[0],
                    y=coords[1],
                    z=coords[2],
                )

                definitions[region_id] = d

        return definitions

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

    def map_regions(self, definitions):
        mapped_regions = []
        regions = self.inputs.regions

        for region_id in regions:
            d = definitions[str(region_id)]

            if not d:
                raise Exception("Missing definition for region id %i." % region_id)

            mapped_regions.append(d)

        return mapped_regions


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
    matrix = traits.Array(
        shape=(None, None),
        desc="the connectivity matrix (calculated using the pearson correlation)"
    )
    p_values = traits.Array(
        shape=(None, None),
        desc="a p-value for each the connectivity matrix element"
    )
    normalized_matrix = traits.Array(
        shape=(None, None),
        desc="the normalized connectivity matrix (Fisher Z transformed)"
    )
    regions = traits.Array(
        shape=(None,),
        desc="the regions represented in the matrix (as defined in the atlas)"
    )


class FunctionalConnectivity(BaseInterface):
    """
    Creates a (functional) connectivity matrix from a fMRI volume using an atlas volume (for defining regions).
    The fMRI volume and the atlas must have the same resolution.
    Each distinct value (!= 0) in the atlas volume is treated as region (0 is air).

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

        results = self.gen_fconn(fmri_data, atlas_data)
        self.matrix = results["matrix"]
        self.p_values = results["p_values"]
        self.normalized_matrix = results["normalized_matrix"]
        self.regions = results["regions"]

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["matrix"] = self.matrix
        outputs["p_values"] = self.p_values
        outputs["normalized_matrix"] = self.normalized_matrix
        outputs["regions"] = self.regions
        return outputs

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
            "regions": regions
        }
