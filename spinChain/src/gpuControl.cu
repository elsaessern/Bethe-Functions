/**
 * @Author: Eric Corwin <ecorwin>
 * @Date:   2017-09-15T11:06:23-07:00
 * @Email:  eric.corwin@gmail.com
 * @Filename: gpuControl.cu
 * @Last modified by:   ecorwin
 * @Last modified time: 2017-09-15T11:07:37-07:00
 */


#include "gpuControl.h"
#include <iostream>
//Pick the gpu, this needs to happen before the packing object gets created
long setDeviceNumber( long deviceNumber ) {

	//Set up the correct gpu
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	//std::cout << __PRETTY_FUNCTION__ << ": num_devices = " << num_devices << std::endl;

	if (deviceNumber < num_devices) {
		cudaSetDevice(deviceNumber);
	}

	int active_device;
	cudaGetDevice(&active_device);
	//std::cout << __PRETTY_FUNCTION__ << ": active_device = " << active_device << std::endl;

	//Tell the cpu to yield while waiting for results from the device
	//cudaSetDeviceFlags(cudaDeviceScheduleYield);
	//TODO: Consider using cudaDeviceScheduleBlockingSync to allow for running without 100% cpu utilization
	return active_device;
}

//Return the active gpu
long getDeviceNumber() {
	int active_device;
	cudaGetDevice(&active_device);
	return active_device;
}
