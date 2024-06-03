import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ai_art_style_transfer_demo/helper/classifier_float.dart';
import 'package:ai_art_style_transfer_demo/helper/transfer_helper.dart';
import 'package:ai_art_style_transfer_demo/helper/transfer_helper_2.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hl_image_picker/hl_image_picker.dart';
import 'package:logger/logger.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter_processing/tflite_flutter_processing.dart';
import 'package:image/image.dart' as img;

import '../helper/transfer_helper_3.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // ClassifierFloat classifierFloat = ClassifierFloat(numThreads: Platform.numberOfPro cessors);
  late final ImageTransferFacade transferHelper ;
  img.Image? styleImage;

  Uint8List? uint8listStyleImage;

  @override
  void initState() {

    super.initState();
    // transferHelper = ImageTransferFacade(
    //     numThreads: Platform.numberOfProcessors,
    //     currentDelegate: TransferHelper.DELEGATE_GPU,
    //     currentModel: 1,
    //     styleTransferListener: StyleTransferListener(
    //         onError: (String error) {
    //           Logger().d('生成失败 :${error}');
    //         },
    //         onResult: onResult));
    transferHelper = ImageTransferFacade();


    WidgetsBinding.instance.addPostFrameCallback((timeStamp) async{
      await transferHelper.loadModel();


      // await transferHelper.init();

      var styleImg = await rootBundle.load('assets/neural_style_transfer_5_1.jpg');
      // var cmd = img.Command()..decodeImage(styleImg.buffer.asUint8List());
      var sImg = img.decodeImage(styleImg.buffer.asUint8List());
      // var cmdExe = await cmd.executeThread();
      styleImage = sImg;
      if(sImg != null){
        Logger().d('init style:${sImg.width}');
        // transferHelper.setStyleImage(sImg);

      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'AI Art Style Transfer Demo',
        ),
      ),
      body: buildBody(),
    );
  }

  Widget buildBody() {
    return SingleChildScrollView(
      child: Column(
        children: [
          Text('selected Image:'),
          if (_selectedImages.firstOrNull != null)
            Image.file(
              File(_selectedImages.first.path),
              width: double.infinity,
              fit: BoxFit.fitWidth,
            ),
          const SizedBox(
            height: 20,
          ),

          const Text('style Image:'),
          if (uint8listStyleImage != null)
            Image.memory(
              uint8listStyleImage!,
              width: double.infinity,
              fit: BoxFit.fitWidth,
            ),
          const SizedBox(
            height: 20,
          ),

          const Text('result:'),
          if (output != null)
            Image.file(
              File(output!),
              width: double.infinity,
              fit: BoxFit.fitWidth,
            ),
          const SizedBox(
            height: 20,
          ),
          ElevatedButton(
            onPressed: chooseImage,
            child: const Text('choose a image'),
          ),
          ElevatedButton(
            onPressed: chooseStyleImage,
            child: const Text('choose a image'),
          ),
          ElevatedButton(
            onPressed: startTransfer,
            child: const Text('start transfer'),
          ),
        ],
      ),
    );
  }

  final _picker = HLImagePicker();

  List<HLPickerItem> _selectedImages = [];
  String? output;

  Future<HLPickerItem?> _openPicker() async {
    final images = await _picker.openPicker(
        // Properties
        pickerOptions: const HLPickerOptions(
            mediaType: MediaType.image,
            maxSelectedAssets: 1,
            minSelectedAssets: 1,
            enablePreview: true));
    return images.firstOrNull;
  }

  void chooseImage() async{
    var p = await _openPicker();
    if(p != null) {
      _selectedImages = [p];
    }
  }

  void startTransfer() async {
    // classifierFloat.loadModel();
    if(_selectedImages.isEmpty){
      return;
    }
    Logger().d('iamge path ${_selectedImages.first.path}');
    var inputImg = await img.decodeImageFile(_selectedImages.first.path);
    if(inputImg == null){
      return;
    }

    // var styleImg = File(_selectedImages.first.path);
    // var cmd = img.Command()..decodeImage(styleImg.readAsBytesSync());
    // var cmdExe = await cmd.executeThread();
    // if(cmdExe.outputImage == null){
    //   return;
    // }
    // transferHelper.transfer(inputImg!);
    var result = await transferHelper.transfer(File(_selectedImages.first.path).readAsBytesSync(),uint8listStyleImage ?? await transferHelper.loadStyleImage('assets/neural_style_transfer_5_1.jpg'));

    var path = await saveFile(result);
    setState(() {
      output = path;
    });

  }

  void onResult(img.Image bitmap, int inferenceTime) async{
    var tempDir = await getTemporaryDirectory();
    if(tempDir.existsSync() == false){
      await tempDir.create();
    }
    var filePath = '${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg';
    // var result = await img.encodeJpgFile(filePath, bitmap);

    Logger().d('style: == null ? ${styleImage == null}');
    if(styleImage != null) {
      var file = File(filePath);
      if(file.existsSync() == false){
        file.createSync();
      }

      await file.writeAsBytes(img.encodeJpg(bitmap));
    }
    // Logger().d('save image r:${result}');
    setState(() {


      output = filePath;
    });



  }

  Future<String> saveFile(Uint8List result) async{
    var tempDir = await getTemporaryDirectory();
    if(tempDir.existsSync() == false){
      await tempDir.create();
    }
    var filePath = '${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg';
    var file = File(filePath);
    if(file.existsSync() == false){
      file.createSync();
    }

    await file.writeAsBytes(result);
    return filePath;
  }

  void chooseStyleImage() async{
    var result = await _openPicker();
    if(result== null){
      return;
    }
    setState(() {
      uint8listStyleImage = File(result.path).readAsBytesSync();
    });

  }
}
