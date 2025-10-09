#!/bin/bash
# Script to download selected Alimama Auto-Bidding Competition datasets to a specified directory

set -e

declare -A URLS
URLS["7-8"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_7-8.zip"
URLS["9-10"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_9-10.zip"
URLS["11-12"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_11-12.zip"
URLS["13"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_13.zip"
URLS["14-15"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_14-15.zip"
URLS["16-17"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_16-17.zip"
URLS["18-19"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_18-19.zip"
URLS["20-21"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_20-21.zip"
URLS["22-23"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_22-23.zip"
URLS["24-25"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_24-25.zip"
URLS["26-27"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_26-27.zip"
URLS["traj1"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_trajectory_data_1.zip"
URLS["traj2"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_trajectory_data_2.zip"
URLS["traj3"]="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_trajectory_data_3.zip"

usage() {
    echo "Usage: $0 -p <periods/trajectories> -d <directory>"
    echo "  -p <periods/trajectories>    Comma-separated list (e.g., 7-8,9-10,traj1,traj2)"
    echo "  -d <directory>               Directory to save downloaded files"
    exit 1
}

while getopts "p:d:" opt; do
    case $opt in
        p) PERIODS="$OPTARG" ;;
        d) DIR="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$PERIODS" || -z "$DIR" ]]; then
    usage
fi

mkdir -p "$DIR"

IFS=',' read -ra SELECTED <<< "$PERIODS"
for period in "${SELECTED[@]}"; do
    url="${URLS[$period]}"
    if [[ -z "$url" ]]; then
        echo "Warning: No URL found for '$period', skipping."
        continue
    fi
    echo "Downloading $period ..."
    wget -c "$url" -P "$DIR"
    zipfile="$DIR/$(basename "$url")"
    if [[ -f "$zipfile" ]]; then
        echo "Unzipping $zipfile ..."
        if [[ "$period" == traj* ]]; then
            # For trajectory files, extract autoBidding_aigb_*.csv files
            unzip -j -o "$zipfile" "autoBidding_aigb_*.csv" -d "$DIR"
        else
            # For general track files, extract period-*.csv files
            unzip -j -o "$zipfile" "period-*.csv" -d "$DIR"
        fi
        rm -f "$zipfile"
    fi
done

echo "Selected downloads and extraction completed."
