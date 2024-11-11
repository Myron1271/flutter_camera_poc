import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  List<dynamic>? _recognitions;
  late Interpreter _interpreter;
  late List<String> _labels;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // Load the model and labels
  Future<void> _loadModel() async {
    // Load the TFLite model
    _interpreter = await Interpreter.fromAsset('model.tflite');
    
    // Load labels
    final labelFile = await File('assets/labels.txt').readAsString();
    _labels = labelFile.split('\n').map((e) => e.trim()).toList();
  }

  // Pick an image from the gallery
  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      _detectObjects(_image!);
    }
  }

  // Perform object detection on the image
  Future<void> _detectObjects(File imageFile) async {
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(Uint8List.fromList(imageBytes));
    
    if (image == null) return;

    // Resize the image to the required input size (assumed to be 224x224 for this example)
    final resizedImage = img.copyResize(image, width: 224, height: 224);

    // Convert the image to a list of pixel values (normalize to [0,1])
    final input = resizedImage.getBytes().map((e) => e / 255.0).toList();

    // Run inference
    final output = List.filled(1 * 10, 0).reshape([1, 10]);
    _interpreter.run(input, output);

    // Process the output (assuming the model has 10 possible classes)
    setState(() {
      _recognitions = List.generate(10, (index) {
        return {
          'label': _labels[index],
          'confidence': output[0][index],
        };
      });
    });
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Object Detection'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: _pickImage,
              child: Text('Add Image'),
            ),
            if (_image != null) ...[
              Image.file(_image!),
              SizedBox(height: 20),
            ],
            if (_recognitions != null) ...[
              for (var recognition in _recognitions!)
                Text(
                  'Label: ${recognition['label']} | Confidence: ${(recognition['confidence'] * 100).toStringAsFixed(2)}%',
                ),
            ],
          ],
        ),
      ),
    );
  }
}
