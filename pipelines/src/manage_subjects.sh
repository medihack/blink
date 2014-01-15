#!/bin/bash

function usage {
  echo "Available options:"
}

list=false
unpack=false

while getopts ":lu" opt; do
  case $opt in
    u)
      unpack=true
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
  echo "-l  Lists subject ids in folder."
  exit 2
fi

cd $1

if $unpack; then
  for file in *.zip; do
    if [[ ${file} =~ "_3T_Structural_preproc" ]]; then
      unzip -n "$file" '*/MNINonLinear/T1w_restore_brain.nii.gz'
      unzip -n "$file" '*/MNINonLinear/brainmask_fs.nii.gz'
    fi

    if [[ ${file} =~ "_3T_rfMRI_REST1_preproc" ]]; then
      unzip -n "$file" '*/MNINonLinear/*/rfMRI_REST1_LR.nii.gz'
    fi
  done
fi

if $list; then
  subject_ids=()

  for file in *.zip; do
    subject_id=${file%%_*}
    subject_ids+=($subject_id)
  done

  echo "${subject_ids[*]}" | tr ' ' '\n' | sort -u
fi
