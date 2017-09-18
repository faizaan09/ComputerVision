// Instructions:
// For question 1, only modify function: histogram_equalization
// For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
// For question 3, only modify function: laplacian_pyramid_blending

#include "./header.h"

using namespace std;
using namespace cv;

void help_message(char* argv[])
{
   cout << "Usage: [Question_Number] [Input_Options] [Output_Options]" << endl;
   cout << "[Question Number]" << endl;
   cout << "1 Histogram equalization" << endl;
   cout << "2 Frequency domain filtering" << endl;
   cout << "3 Laplacian pyramid blending" << endl;
   cout << "[Input_Options]" << endl;
   cout << "Path to the input images" << endl;
   cout << "[Output_Options]" << endl;
   cout << "Output directory" << endl;
   cout << "Example usages:" << endl;
   cout << argv[0] << " 1 " << "[path to input image] " << "[output directory]" << endl;
   cout << argv[0] << " 2 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
   cout << argv[0] << " 3 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
}

// ===================================================
// ======== Question 1: Histogram equalization =======
// ===================================================

Mat histogram_equalization(const Mat& img_in)
{
   // Write histogram equalization codes here
   Mat img_out = img_in; // Histogram equalization result

   return img_out;
}

bool Question1(char* argv[])
{
   // Read in input images
   Mat input_image = imread(argv[2], IMREAD_COLOR);

   // Histogram equalization
   Mat output_image = histogram_equalization(input_image);

   // Write out the result
   string output_name = string(argv[3]) + string("1.jpg");
   imwrite(output_name.c_str(), output_image);

   return true;
}

// ===================================================
// ===== Question 2: Frequency domain filtering ======
// ===================================================

Mat low_pass_filter(const Mat& img_in)
{
   // Write low pass filter codes here
   Mat img_out = img_in; // Low pass filter result

   return img_out;
}

Mat high_pass_filter(const Mat& img_in)
{
   // Write high pass filter codes here
   Mat img_out = img_in; // High pass filter result

   return img_out;
}

Mat deconvolution(const Mat& img_in)
{
   // Write deconvolution codes here
   Mat img_out = img_in; // Deconvolution result

   return img_out;
}

bool Question2(char* argv[])
{
   // Read in input images
   Mat input_image1 = imread(argv[2], IMREAD_COLOR);
   Mat input_image2 = imread(argv[3], IMREAD_COLOR);

   // Low and high pass filters
   Mat output_image1 = low_pass_filter(input_image1);
   Mat output_image2 = high_pass_filter(input_image1);

   // Deconvolution
   Mat output_image3 = deconvolution(input_image2);

   // Write out the result
   string output_name1 = string(argv[4]) + string("2.jpg");
   string output_name2 = string(argv[4]) + string("3.jpg");
   string output_name3 = string(argv[4]) + string("4.jpg");
   imwrite(output_name1.c_str(), output_image1);
   imwrite(output_name2.c_str(), output_image2);
   imwrite(output_name3.c_str(), output_image3);

   return true;
}

// ===================================================
// ===== Question 3: Laplacian pyramid blending ======
// ===================================================

Mat laplacian_pyramid_blending(const Mat& img_in1, const Mat& img_in2)
{
   // Write laplacian pyramid blending codes here
   Mat img_out = img_in1; // Blending result

   return img_out;
}

bool Question3(char* argv[])
{
   // Read in input images
   Mat input_image1 = imread(argv[2], IMREAD_COLOR);
   Mat input_image2 = imread(argv[3], IMREAD_COLOR);

   // Histogram equalization
   Mat output_image = laplacian_pyramid_blending(input_image1, input_image2);

   // Write out the result
   string output_name = string(argv[4]) + string("5.jpg");
   imwrite(output_name.c_str(), output_image);

   return true;
}

int main(int argc, char* argv[])
{
   int question_number = -1;

   // Validate the input arguments
   if (argc < 4) {
      help_message(argv);
      exit(1);
   }
   else {
      question_number = atoi(argv[1]);

      if (question_number == 1 && !(argc == 4)) {
         help_message(argv);
         exit(1);
      }
      if (question_number == 2 && !(argc == 5)) {
	 help_message(argv);
	 exit(1);
      }
      if (question_number == 3 && !(argc == 5)) {
	 help_message(argv);
	 exit(1);
      }
      if (question_number > 3 || question_number < 1 || argc > 5) {
	 cout << "Input parameters out of bound ..." << endl;
	 exit(1);
      }
   }

   switch (question_number) {
      case 1: Question1(argv); break;
      case 2: Question2(argv); break;
      case 3: Question3(argv); break;
   }

   return 0;
}
