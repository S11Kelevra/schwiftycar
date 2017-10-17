#!/bin/sh
cd /Users/robotics/robocar42
echo `pwd`
if [ "$(/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I | awk '/ SSID/ {print substr($0, index($0, $2))}')" != 'robotics' ]; then
	echo 'Computer is not connected to the "robotics" wifi'
	exit
else
	echo 'Connected to "robotics" wifi'
fi

echo "Attempting to communicate with car"
python /Users/robotics/robocar42/scripts/drive.py

read -sn 1 -p "Press any key to continue..."
