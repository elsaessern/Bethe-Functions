/**
 * @Author: Eric Corwin <ecorwin>
 * @Date:   2017-09-15T11:05:37-07:00
 * @Email:  eric.corwin@gmail.com
 * @Filename: gpuControl.h
 * @Last modified by:   ecorwin
 * @Last modified time: 2018-05-23T12:01:02-07:00
 */

#ifndef GPUCONTROL_H_
#define GPUCONTROL_H_

//Pick the gpu, this needs to happen before the packing object gets created
long setDeviceNumber( long deviceNumber );

//Return the active gpu
long getDeviceNumber();

#endif /* GPUCONTROL_H_ */
