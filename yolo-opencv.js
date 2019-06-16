const cv = require('opencv4nodejs');
const path = require('path');
const fs = require('fs');

configPath = '/home/shekar/Downloads/pan/yolo-object-detection/yolo-coco/yolov3.cfg';
weightsPath = '/home/shekar/Downloads/pan/yolo-object-detection/yolo-coco/yolov3.weights';

const darknetPath = "/home/shekar/Downloads/pan/yolo-object-detection/yolo-coco";

const cfgFile = path.resolve(darknetPath, "yolov3.cfg");
const weightsFile = path.resolve(darknetPath, "yolov3.weights");
const labelsFile = path.resolve(darknetPath, "coco.names");


const labels = fs
    .readFileSync(labelsFile)
    .toString()
    .split("\n");

const net = cv.readNetFromDarknet(cfgFile, weightsFile);

const allLayerNames = net.getLayerNames();
const unconnectedOutLayers = net.getUnconnectedOutLayers();

const layerNames = unconnectedOutLayers.map(layerIndex => {
    return allLayerNames[layerIndex - 1];
});
// console.log(allLayerNames);
console.log(layerNames);


var image = cv.imread("/home/shekar/Downloads/pan/yolo-object-detection/images/dining_table.jpg")
const minConfidence = 0.6;
const nmsThreshold = 0.3;




const size = new cv.Size(416, 416);
const vec3 = new cv.Vec(0, 0, 0);
const [imgHeight, imgWidth] = image.sizes;

// network accepts blobs as input
const inputBlob = cv.blobFromImage(image, 1 / 255.0, size, vec3, true, false);
net.setInput(inputBlob);

console.time("net.forward");
// forward pass input through entire network
const layerOutputs = net.forward(layerNames);
console.timeEnd("net.forward");

// console.log(layerOutputs[0].getDataAsArray()[0].slice(5))

layerOutputs.forEach(mat => {
    const output = mat.getDataAsArray();

    output.forEach(detection => {
        var confidences = detection.slice(5);
        var classId = confidences.indexOf(Math.max(...confidences));
        var confidance = confidences[classId];
        if (confidance > minConfidence) {

            const box = detection.slice(0, 4);

            const centerX = parseInt(box[0] * imgWidth);
            const centerY = parseInt(box[1] * imgHeight);
            const width = parseInt(box[2] * imgWidth);
            const height = parseInt(box[3] * imgHeight);

            const x = parseInt(centerX - width / 2);
            const y = parseInt(centerY - height / 2);
            console.log(labels[classId] + ': ' + confidance)
        }
    });

});



