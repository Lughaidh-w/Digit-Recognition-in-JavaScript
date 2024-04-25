const tf = require('@tensorflow/tfjs-node');
const mnist = require('mnist');
const fs = require('fs');

var set = mnist.set(8000, 2000);

var trainingSet = set.training;
var testSet = set.test;


const modelDir = './model';
if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir);
}


const trainingFeatures = trainingSet.map(item => item.input);
const trainingLabels = trainingSet.map(item => item.output);
const testFeatures = testSet.map(item => item.input);
const testLabels = testSet.map(item => item.output);


const trainingFeaturesTensor = tf.tensor2d(trainingFeatures);
const trainingLabelsTensor = tf.tensor2d(trainingLabels);
const testFeaturesTensor = tf.tensor2d(testFeatures);
const testLabelsTensor = tf.tensor2d(testLabels);


const model = tf.sequential();
model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [784]}));
model.add(tf.layers.dense({units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ["accuracy"]});

async function trainAndSaveModel() {
    // Train the model
    await model.fit(trainingFeaturesTensor, trainingLabelsTensor, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });

    // Save the model
    await model.save(`file://${modelDir}`);
}


trainAndSaveModel();