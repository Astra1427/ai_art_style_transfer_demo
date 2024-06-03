

import 'package:tflite_flutter_processing/tflite_flutter_processing.dart';

import 'classifier.dart';

class ClassifierFloat extends Classifier {
  ClassifierFloat({super.numThreads});

  @override
  String get modelName => 'mobilenet_v1_1.0_224.tflite';

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(127.5, 127.5);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);
}