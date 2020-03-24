#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// загрузка чёрно белых изображений с метками из csv файла
class DataLoader {
	int width; // ширина изображения
	int height; // высота изображения
	vector<string> labels; // метки

	vector<string> SplitLine(const string &line, char delimeter = ',') const; // разбиение строки по разделителю
	vector<double> PixelsToVector(const vector<string> &values) const; // получение вектора картинки
	vector<double> LabelToVector(const string &label) const; // получение вектора метки
public:
	DataLoader(const string &path); // создание загрузчика
	NetworkData LoadData(const string &path); // считывание данных
};

// создание загрузчика
DataLoader::DataLoader(const string &path) {
	ifstream f(path); // пытаемся открыть файл

	// если не удалось
	if (!f)
		throw runtime_error("invalid file"); // бросаем исключение
	
	string sizesStr; // строка с размерами
	string labelsStr; // строка с метками
	getline(f, sizesStr); // считываем размеры
	getline(f, labelsStr); // считываем метки
	f.close(); // закрываем файл

	vector<string> sizes = SplitLine(sizesStr, ' '); // разбиваем размеры

	width = stoi(sizes[0]); // получаем ширину
	height = stoi(sizes[1]); // получаем высоту
	labels = SplitLine(labelsStr, ' '); // получаем метки

	cout << "sizes: [" << width << " x " << height << "]" << endl; // выводим размеры
	cout << "labels: " << labels.size() << endl; // выводим число меток
}

// разбиение строки по пробелам
vector<string> DataLoader::SplitLine(const string &line, char delimeter) const {
	vector<string> values;
	string s = "";

	for (int i = 0; i < line.length(); i++) {
		if (line[i] == delimeter) {
			values.push_back(s);
			s = "";
		}
		else {
			s += line[i];
		}
	}

	if (s != "")
		values.push_back(s);

	return values;
}

// получение вектора из пикселей
vector<double> DataLoader::PixelsToVector(const vector<string> &values) const {
	vector<double> x(width * height, 0);

	for (int i = 1; i < values.size(); i++)
		x[i - 1] = stod(values[i]) / 255.0;

	return x;
}

// получение вектора из метки
vector<double> DataLoader::LabelToVector(const string &label) const {
	vector<double> y(labels.size(), 0);

	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] == label) {
			y[i] = 1;
			return y;
		}
	}

	throw runtime_error("invalid label");
}

// загрузка данных
NetworkData DataLoader::LoadData(const string &path) {
	ifstream f(path); // пытаемся открыть файл

	// если не получилось
	if (!f)
		throw runtime_error("invalid file with data"); // бросаем исключение

	NetworkData data; // данные
	string line; // строка для считывания
	getline(f, line); // считываем строку заголовок

	// пока есть строки
	while (getline(f, line)) {
		vector<string> values = SplitLine(line); // разбиваем строку по запятой

		// если количество некорректно
		if (values.size() != 1 + width * height)
			throw runtime_error("invalid line"); // бросаем исключение

		data.x.push_back(PixelsToVector(values)); // добавляем вектор изображения
		data.y.push_back(LabelToVector(values[0])); // добавляем вектор метки
		cout << "load " << data.x.size() << " lines\r"; // выводим, сколько уже строк обработано
	}

	f.close(); // закрываем файл
	cout << "load " << data.x.size() << " values" << endl; // выводим число считанных строк

	return data; // возвращаем считанные данные
}