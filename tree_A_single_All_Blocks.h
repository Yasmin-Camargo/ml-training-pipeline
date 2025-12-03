
/*
This inline function was automatically generated using DecisionTreeToCpp Converter

It takes feature vector as single argument:
feature_vector[0] - frame
feature_vector[1] - x
feature_vector[2] - y
feature_vector[3] - Width
feature_vector[4] - Height
feature_vector[5] - BlockSize
feature_vector[6] - Area
feature_vector[7] - splitSeries
feature_vector[8] - cuQP
feature_vector[9] - IntraMode
feature_vector[10] - AbsSumResidual
feature_vector[11] - AbsSumUltimaLinha
feature_vector[12] - AbsSumUltimaColuna
feature_vector[13] - lefTopResidual
feature_vector[14] - leftBottomResidual
feature_vector[15] - rightTopResidual
feature_vector[16] - rightBottomResidual
feature_vector[17] - DCT2_DCT2cost


It returns index of predicted class:
0 - 10
1 - 20


Simply include this file to your project and use it
*/

#include <vector>

inline int tree_A_single_All_Blocks(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(10) <= 35.5) {
		if (feature_vector.at(17) <= 4.999999921591263e+17) {
			if (feature_vector.at(7) <= 71402528.0) {
				return 0;
			}
			else {
				return 0;
			}
		}
		else {
			return 1;
		}
	}
	else {
		if (feature_vector.at(17) <= 4.999999935734331e+17) {
			if (feature_vector.at(17) <= 31971794.0) {
				if (feature_vector.at(11) <= 200.5) {
					if (feature_vector.at(6) <= 48.0) {
						if (feature_vector.at(10) <= 334.5) {
							if (feature_vector.at(0) <= 115.5) {
								return 0;
							}
							else {
								if (feature_vector.at(9) <= 28.5) {
									return 1;
								}
								else {
									return 1;
								}
							}
						}
						else {
							if (feature_vector.at(13) <= 0.5) {
								return 1;
							}
							else {
								return 1;
							}
						}
					}
					else {
						if (feature_vector.at(5) <= 12.0) {
							return 0;
						}
						else {
							if (feature_vector.at(12) <= 65.5) {
								if (feature_vector.at(6) <= 96.0) {
									return 0;
								}
								else {
									if (feature_vector.at(15) <= -0.5) {
										return 1;
									}
									else {
										if (feature_vector.at(6) <= 192.0) {
											return 1;
										}
										else {
											return 1;
										}
									}
								}
							}
							else {
								if (feature_vector.at(17) <= 2382364.5) {
									if (feature_vector.at(16) <= -4.5) {
										return 1;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(14) <= 6.5) {
										if (feature_vector.at(10) <= 1612.5) {
											if (feature_vector.at(14) <= 1.5) {
												if (feature_vector.at(3) <= 6.0) {
													return 0;
												}
												else {
													return 0;
												}
											}
											else {
												return 1;
											}
										}
										else {
											if (feature_vector.at(13) <= -1.5) {
												return 0;
											}
											else {
												return 1;
											}
										}
									}
									else {
										return 0;
									}
								}
							}
						}
					}
				}
				else {
					if (feature_vector.at(10) <= 2215.0) {
						if (feature_vector.at(7) <= 2215457.0) {
							if (feature_vector.at(0) <= 146.0) {
								return 1;
							}
							else {
								return 0;
							}
						}
						else {
							if (feature_vector.at(16) <= -4.5) {
								return 1;
							}
							else {
								return 1;
							}
						}
					}
					else {
						if (feature_vector.at(12) <= 750.0) {
							if (feature_vector.at(2) <= 1526.0) {
								if (feature_vector.at(9) <= 32.0) {
									if (feature_vector.at(9) <= 0.5) {
										return 1;
									}
									else {
										return 1;
									}
								}
								else {
									return 1;
								}
							}
							else {
								return 1;
							}
						}
						else {
							return 1;
						}
					}
				}
			}
			else {
				return 0;
			}
		}
		else {
			return 1;
		}
	}
}