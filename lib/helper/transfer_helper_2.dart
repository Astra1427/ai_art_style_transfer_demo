import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:ai_art_style_transfer_demo/helper/transfer_helper.dart';
import 'package:logger/logger.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:ui' as ui;
import 'package:image/image.dart';
import 'package:tflite_flutter_processing/tflite_flutter_processing.dart';

class TransferHelper2 {
  static const DELEGATE_CPU = 0;
  static const DELEGATE_GPU = 1;
  static const DELEGATE_NNAPI = 2;
  static const MODEL_INT8 = 0;

  static const TAG = "Style Transfer Helper";

  Interpreter? interpreterPredict;
  Interpreter? interpreterTransform;
  Image? styleImage;

  var inputPredictTargetWidth = 0;
  var inputPredictTargetHeight = 0;
  var inputTransformTargetWidth = 0;
  var inputTransformTargetHeight = 0;
  var outputPredictShape = <int>[];
  var outputTransformShape = <int>[];

  final int numThreads;
  final int currentDelegate;
  final int currentModel;
  final StyleTransferListener styleTransferListener;

  TransferHelper2({
    required this.numThreads,
    required this.currentDelegate,
    required this.currentModel,
    required this.styleTransferListener,
  });

  Future init() async {
    if (await setupStyleTransfer()) {
      inputPredictTargetHeight = interpreterPredict!.getInputTensor(0).shape[1];
      inputPredictTargetWidth = interpreterPredict!.getInputTensor(0).shape[2];
      outputPredictShape = interpreterPredict!.getOutputTensor(0).shape;

      inputTransformTargetHeight =
      interpreterTransform!.getInputTensor(0).shape[1];
      inputTransformTargetWidth =
      interpreterTransform!.getInputTensor(0).shape[2];
      outputTransformShape = interpreterTransform!.getOutputTensor(0).shape;
    } else {
      styleTransferListener.onError("TFLite failed to init.");
    }
  }

  Future<bool> setupStyleTransfer() async {
    var tfliteOption = InterpreterOptions();
    tfliteOption.threads = numThreads;

    switch (currentDelegate) {
      case DELEGATE_CPU:
        break;
      case DELEGATE_GPU:
        try {
          if (Platform.isIOS) {
            tfliteOption.addDelegate(GpuDelegateV2());
          } else {
            tfliteOption.addDelegate(XNNPackDelegate());
          }
        } catch (e, s) {
          styleTransferListener
              .onError("GPU is not supported on this device,\n${e},\n${s}");
        }
        break;
      case DELEGATE_NNAPI:
      // tfliteOption.addDelegate(NnApiDelegate());
        break;
    }

    String modelPredict;
    String modelTransfer;

    if (currentModel == MODEL_INT8) {
      modelPredict = "assets/models/predict_int8.tflite";
      modelTransfer = "assets/models/transfer_int8.tflite";
    } else {
      modelPredict = "assets/models/predict_float16.tflite";
      modelTransfer = "assets/models/transfer_float16.tflite";
    }

    try {
      interpreterPredict =
      await Interpreter.fromAsset(modelPredict, options: tfliteOption);
      interpreterTransform =
      await Interpreter.fromAsset(modelTransfer, options: tfliteOption);
      return true;
    } catch (e, s) {
      styleTransferListener.onError(
          "Style transfer failed to initialize. See error logs for details\n${e}");
      return false;
    }
  }

  void transfer(Image bitmap) {
    if (interpreterPredict == null || interpreterTransform == null) {
      setupStyleTransfer();
    }

    if (styleImage == null) {
      styleTransferListener
          .onError("Please select the style before run the transforming");
      return;
    }

    var inferenceTime = DateTime.now().millisecondsSinceEpoch;

    var inputImage = processInputImage(
        image: bitmap,
        targetWidth: inputTransformTargetWidth,
        targetHeight: inputTransformTargetHeight);
    if (inputImage == null) {
      styleTransferListener.onError("image == null");
      return;
    }

    var sImage = processInputImage(
        image: styleImage!,
        targetWidth: inputPredictTargetWidth,
        targetHeight: inputPredictTargetHeight);
    if (sImage == null) {
      styleTransferListener.onError("sImage == null");
      return;
    }

    var predictOutput =
    TensorBuffer.createFixedSize(outputPredictShape, TensorType.float32);
    interpreterPredict?.run(sImage.buffer, predictOutput.buffer);

    var transformInput = [inputImage.buffer, predictOutput.buffer];

    var outputImage =
    TensorBuffer.createFixedSize(outputTransformShape, TensorType.float32);
    interpreterTransform?.runForMultipleInputs(
        transformInput, <int, Object>{0: outputImage.buffer});

    Logger().d('shape :${outputImage.getShape()}');
    var oLength = outputImage.getShape().length;
    var oWidth = oLength == 3 ? outputImage.getShape()[0] : outputImage.getShape()[1];
    var oHeight = oLength == 3 ? outputImage.getShape()[1] : outputImage.getShape()[2];
    var outputBitmap = getOutputImage(outputImage, oWidth, oHeight);
    if (outputBitmap == null) {
      styleTransferListener.onError('result == null');
      return;
    }

    inferenceTime = DateTime.now().millisecondsSinceEpoch - inferenceTime;
    styleTransferListener.onResult(outputBitmap, inferenceTime);
  }

  void setStyleImage(Image bitmap) {
    styleImage = bitmap;
  }

  void clearStyleTransferHelper() {
    interpreterPredict = null;
    interpreterTransform = null;
  }

  TensorImage? processInputImage({
    required Image image,
    required int targetWidth,
    required int targetHeight,
  }) {
    var height = image.height;
    var width = image.width;
    var cropSize = min(height, width);

    var imageProcessor = ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(targetHeight, targetWidth, ResizeMethod.BILINEAR))
        .add(NormalizeOp(0.0, 1.0)) // Normalize to [0, 1] range
        .build();

    var tensorImage = TensorImage(TensorType.float32);
    tensorImage.loadImage(image);
    return imageProcessor.process(tensorImage);
  }

  Image? getOutputImage(TensorBuffer output, int width, int height) {
    var imagePostProcessor =
    ImageProcessorBuilder().add(DequantizeOp(0.0, 1.0)).build();
    var tensorImage = TensorImage(TensorType.float32);
    tensorImage.loadTensorBuffer(output);
    return tensorImage.image;
  }
}
