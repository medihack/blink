#!/bin/bash

# Inputs
# aal.nii.gz, ch2.nii.gz from mricron
# MNI152_T1_1mm_brain.nii.gz from FSL
# rfMRI_REST1_LR from a HCP subject

fslorient -swaporient ch2
fslswapdim ch2 -x y z ch2

fslorient -swaporient aal
fslswapdim aal -x y z aal

bet ch2 ch2_brain -R

flirt -in ch2_brain -ref MNI152_T1_1mm_brain -out ch2mni_aff -omat ch2mni_aff.mat
fnirt --in=ch2_brain --ref=MNI152_T1_1mm_brain --aff=ch2mni_aff.mat --iout=ch2mni_naff --cout=ch2mni_warp

applywarp --in=aal --ref=MNI152_T1_1mm_brain --interp=nn --warp=ch2mni_warp --out=aal2mni
applywarp --in=aal --ref=rfMRI_REST1_LR --interp=nn --warp=ch2mni_warp --out=aal2mni_2mm
