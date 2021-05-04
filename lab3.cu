/**
 * @file lab3.cpp
 * @author Abhishek Bhaumick (abhaumic@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2021-01-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include "lab3.cuh"
#include "cpuLib.h"
#include "cudaLib.cuh"






int main(int argc, char** argv) {
	int batchSize = 8;
	AlexNet(batchSize);
	
	return 0;
}



