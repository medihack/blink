#!/bin/bash

function usage {
    echo "Available options:"
}

list=false
zips=false
unpack=false

while getopts ":ulz" opt; do
    case $opt in
        u)
            unpack=true
            ;;
        z)
            zips=true
            ;;
        l)
            list=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            ;;
    esac
done
shift $((OPTIND-1))

if [ "$#" -ne 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 [options] path_to_subjects_folder"
    echo "-u  Unpack subjects in folder (skips already unpacked subjects)."
    echo "-l  Lists unpacked subject ids in provided folder."
    echo "-z  Lists zipped subject ids in provided folder."
    exit 2
fi

cd $1

if $unpack; then
    for file in *.zip; do
        if [[ ${file} =~ "_3T_Structural_preproc" ]]; then
            unzip -n "$file" '*/MNINonLinear/T1w_restore_brain.nii.gz'
            unzip -n "$file" '*/MNINonLinear/brainmask_fs.nii.gz'
            unzip -n "$file" '*/MNINonLinear/xfms/standard2acpc_dc.nii.gz'
        fi

        if [[ ${file} =~ "_3T_rfMRI_REST1_preproc" ]]; then
            unzip -n "$file" '*/MNINonLinear/*/rfMRI_REST1_LR.nii.gz'
        fi

        if [[ ${file} =~ "_3T_Diffusion_preproc" ]]; then
            unzip -n "$file" '*/T1w/ribbon.nii.gz'
            unzip -n "$file" '*/T1w/Diffusion/data.nii.gz'
            unzip -n "$file" '*/T1w/Diffusion/nodif_brain_mask.nii.gz'
            unzip -n "$file" '*/T1w/Diffusion/bvals'
            unzip -n "$file" '*/T1w/Diffusion/bvecs'
        fi
    done
fi

if $list; then
    subject_ids=()
    for folder in `find . -maxdepth 1 -type d`; do
        if [[ $folder =~ \./[0-9]{6}$ ]]; then
            subject_id=${folder##./}
            subject_ids+=($subject_id)
        fi
    done

    echo "${subject_ids[*]}" | tr ' ' '\n' | sort -u
fi

if $zips; then
    subject_ids=()

    for file in *.zip; do
        subject_id=${file%%_*}
        subject_ids+=($subject_id)
    done

    echo "${subject_ids[*]}" | tr ' ' '\n' | sort -u
fi
