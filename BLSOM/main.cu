#include<iostream>
#include"BLSOM.h"
#include"SelectGPU.h"
#include"LoadDataSet.h"
#include<curand_kernel.h>
#include<algorithm>

#define MAP_WIDTH 200
#define MAP_HEIGHT 50
#define TRAIN_NUM 200
#define EPOC_NUM 0

int WriteSOMMAP(std::string fileName, float* map, int map_vec, int map_width, int map_height) {
	std::ofstream ofs;
	ofs.open(fileName, 'w');

	if (!ofs) {
		std::cerr << "can't opne file" << std::endl;
		return EXIT_FAILURE;
	}

	ofs << map_vec << std::endl;
	ofs << map_width << std::endl;
	ofs << map_height << std::endl;

	for (int i = 1; i < map_height*map_width; i++) {
		for (int v = 0; v < map_vec; v++) {
			ofs << *map << " ";
			map++;
		}
		ofs << "\n";
	}
	ofs.close();

	return EXIT_SUCCESS;
}

int WriteUmatrix(std::string fileName, std::vector<std::vector<float>> umatrix) {
	std::ofstream ofs;
	ofs.open(fileName, 'w');

	if (!ofs) {
		std::cerr << "can't opne file" << std::endl;
		return EXIT_FAILURE;
	}

	for (int h = 0; h < umatrix.size()-1; h++) {
		for (int w = 0; w < umatrix[0].size()-1; w++) {
			ofs << umatrix[h][w];
			if (w != umatrix[0].size() - 2)
				ofs << "\t";
		}
		if (h != umatrix.size() - 2)
			ofs << "\n";
	}
	ofs.close();

	return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
	int device;
	int vec_dim;
	int map_width;
	int map_height;
	float* som;
	std::vector<std::vector<float>> umatrix;

	std::shared_ptr<float> map_weight;
	std::vector<std::vector<float>> train;
	std::vector<std::vector<std::vector<float>>> epocs;

	std::vector<float> ave_vec;
	std::vector<std::vector<float>> rotation;
	std::vector<float> sdev;

	train = LoadTrains("sample\\train\\convImg2Txt.txt",' ');
	ave_vec = LoadAverageVector("sample\\train\\average_vector.txt");
	rotation = LoadRotation("sample\\train\\rotation.txt");
	sdev = LoadStandardDev("sample\\train\\sdev.txt");


	map_width = MAP_WIDTH;
	map_height = MAP_HEIGHT;
	vec_dim = ave_vec.size();

	BLSOM blsom = BLSOM(vec_dim, map_width);
	blsom.Init(sdev[0], sdev[1], rotation[0].data(), rotation[1].data(), ave_vec.data());
	blsom.SetTrainingData(train);
	blsom.InitMapWeight(INIT_BATCH);

	/* Get initial map */
	som = blsom.GetSOMMap();
	WriteSOMMAP("sample\\result\\init_batch_map.txt", som, vec_dim, map_width, blsom.MapHeight());

	/* Get initial umatrix */
	umatrix = blsom.GetUMatrix();
	WriteUmatrix("sample\\result\\init_umatrix.txt", umatrix);


	/* Learning */
	blsom.Learning(50);

	/* Get Learned Map */
	som = blsom.GetSOMMap();
	WriteSOMMAP("sample\\result\\result_batch_map.txt", som, vec_dim, map_width, blsom.MapHeight());

	/* Get Umatrix */
	umatrix = blsom.GetUMatrix();
	WriteUmatrix("sample\\result\\result_umatrix.txt", umatrix);

	return 0;
}