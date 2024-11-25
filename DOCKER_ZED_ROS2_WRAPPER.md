## Vanila Setup
Tested but doesn't work:
```bash
./jetson_build_dockerfile_from_sdk_and_l4T_version.sh l4t-r36.3.0 zedsdk4.1.2
```

When running rosdep, the dependencies can't be found in apt repository. 
Problem finding these packages in apt repo:
- ARG GEOGRAPHIC_INFO_VERSION=1.0.4
- ARG ROBOT_LOCALIZATION_VERSION=3.4.2

