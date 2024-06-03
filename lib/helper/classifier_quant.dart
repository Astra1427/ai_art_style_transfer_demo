import 'package:tflite_flutter_processing/tflite_flutter_processing.dart';

import 'classifier.dart';

class ClassifierQuant extends Classifier {
  ClassifierQuant({super.numThreads = 1});

  @override
  String get modelName => 'mobilenet_v1_1.0_224_quant.tflite';

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 255);
}
