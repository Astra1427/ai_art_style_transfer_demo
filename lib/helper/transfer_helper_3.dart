import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ImageTransferFacade {
  // final _predictionModelFile = 'assets/models/predict_int8.tflite';
  // final _transformModelFile = 'assets/models/transfer_int8.tflite';
  final _predictionModelFile = 'assets/models/fp16_predict_v3.tflite';
  final _transformModelFile = 'assets/models/fp16_transfer_v3.tflite';

  // static const int MODEL_TRANSFER_IMAGE_SIZE = 384;
  // static const int MODEL_PREDICTION_IMAGE_SIZE = 256;

  var inputPredictTargetWidth = 0;
  var inputPredictTargetHeight = 0;
  var inputTransformTargetWidth = 0;
  var inputTransformTargetHeight = 0;

  late Interpreter interpreterPrediction;
  late Interpreter interpreterTransform;

  Future<void> loadModel() async {
    // TODO Exception
    interpreterPrediction = await Interpreter.fromAsset(_predictionModelFile);
    interpreterTransform = await Interpreter.fromAsset(_transformModelFile);

    inputPredictTargetHeight = interpreterPrediction!.getInputTensor(0).shape[1];

    inputPredictTargetWidth = interpreterPrediction!.getInputTensor(0).shape[2];

    inputTransformTargetHeight =
    interpreterTransform!.getInputTensor(0).shape[1];
    inputTransformTargetWidth =
    interpreterTransform!.getInputTensor(0).shape[2];
  }

  Future<Uint8List> loadStyleImage(String styleImagePath) async {
    var styleImageByteData = await rootBundle.load(styleImagePath);
    return styleImageByteData.buffer.asUint8List();
  }

  Future<Uint8List> transfer(Uint8List originData, Uint8List styleData) async {
    var originImage = img.decodeImage(originData);
    var modelTransferImage = img.copyResize(originImage!,
        width: inputTransformTargetWidth, height: inputTransformTargetHeight);
    var modelTransferInput = _imageToByteListUInt8(
        modelTransferImage, inputTransformTargetWidth,inputTransformTargetHeight, 0, 255);

    var styleImage = img.decodeImage(styleData);

    // style_image 256 256 3
    var modelPredictionImage = img.copyResize(styleImage!,
        width: inputPredictTargetWidth,
        height: inputPredictTargetHeight);

    // content_image 384 384 3
    var modelPredictionInput = _imageToByteListUInt8(
        modelPredictionImage, inputPredictTargetWidth,inputPredictTargetHeight, 0, 255);

    // style_image 1 256 256 3
    var inputsForPrediction = [modelPredictionInput];
    // style_bottleneck 1 1 100
    var outputsForPrediction = <int, Object>{};
    var styleBottleneck = [
      [
        [List.generate(100, (index) => 0.0)]
      ]
    ];
    outputsForPrediction[0] = styleBottleneck;

    // style predict model
    interpreterPrediction.runForMultipleInputs(
        inputsForPrediction, outputsForPrediction);

    // content_image + styleBottleneck
    var inputsForStyleTransfer = [modelTransferInput, styleBottleneck];

    var outputsForStyleTransfer = <int, Object>{};
    // stylized_image 1 384 384 3
    var outputImageData = [
      List.generate(
        inputTransformTargetWidth,
            (index) => List.generate(
              inputTransformTargetHeight,
              (index) => List.generate(3, (index) => 0.0),
        ),
      ),
    ];
    outputsForStyleTransfer[0] = outputImageData;

    interpreterTransform.runForMultipleInputs(
        inputsForStyleTransfer, outputsForStyleTransfer);

    var outputImage = _convertArrayToImage(outputImageData, inputTransformTargetWidth,inputTransformTargetHeight);
    var rotateOutputImage = img.copyRotate(outputImage, angle:90);
    var flipOutputImage = img.flipHorizontal(rotateOutputImage);
    var resultImage = img.copyResize(flipOutputImage, width: originImage.width, height: originImage.height);
    return img.encodeJpg(resultImage);
  }

  img.Image _convertArrayToImage(List<List<List<List<double>>>> imageArray, int width,int height) {
    img.Image image = img.Image(width: width,height:  height);
    for (var x = 0; x < imageArray[0].length; x++) {
      for (var y = 0; y < imageArray[0][0].length; y++) {
        var r = (imageArray[0][x][y][0] * 255).toInt();
        var g = (imageArray[0][x][y][1] * 255).toInt();
        var b = (imageArray[0][x][y][2] * 255).toInt();
        image.setPixelRgba(x, y, r, g, b, 1);
      }
    }
    return image;
  }

  /*Uint8List _imageToByteListUInt8(
      img.Image image,
      int inputSize,
      double mean,
      double std,
      ) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        img.get
        buffer[pixelIndex++] = (img.getRed(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getGreen(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getBlue(pixel) - mean) / std;
      }
    }
    return convertedBytes.buffer.asUint8List();
  }*/

  Uint8List _imageToByteListUInt8(img.Image image, int width,int height, double mean, double std) {
    var convertedBytes = Float32List(1 * width * height * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < width; i++) {
      for (var j = 0; j < height; j++) {
        final pixel = image.getPixel(j, i); // returns a Pixel object
        buffer[pixelIndex++] = (pixel.r - mean) / std; // Pixel object has a getter for each channel
        buffer[pixelIndex++] = (pixel.g - mean) / std;
        buffer[pixelIndex++] = (pixel.b - mean) / std;
      }
    }
    return convertedBytes.buffer.asUint8List();
  }
}