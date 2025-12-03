
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
feature_vector[7] - depth
feature_vector[8] - CurrQP
feature_vector[9] - splitSeries
feature_vector[10] - BitDepth
feature_vector[11] - cuQP
feature_vector[12] - isMIP
feature_vector[13] - IntraMode
feature_vector[14] - multiRefIdx
feature_vector[15] - AbsSumResidual
feature_vector[16] - AbsSumUltimaLinha
feature_vector[17] - AbsSumUltimaColuna
feature_vector[18] - lefTopResidual
feature_vector[19] - leftBottomResidual
feature_vector[20] - rightTopResidual
feature_vector[21] - rightBottomResidual
feature_vector[22] - DCT2_DCT2cost
feature_vector[23] - BorderContact
feature_vector[24] - FrameWidth
feature_vector[25] - FrameHeight


It returns index of predicted class:
0 - 10
1 - 20


Simply include this file to your project and use it
*/

#include <vector>

inline int tree_B_single_All_Blocks(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(22) <= 95769868.0) {
		if (feature_vector.at(6) <= 192.0) {
			if (feature_vector.at(13) <= 34.5) {
				if (feature_vector.at(0) <= 215.5) {
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
			if (feature_vector.at(3) <= 24.0) {
				return 1;
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