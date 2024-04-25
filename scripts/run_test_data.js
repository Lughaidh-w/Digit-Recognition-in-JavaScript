const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const getMnistData = require('./mnist_data.js');


const modelDir = './model';
if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir);
}

const { trainingFeatures, trainingLabels, 
    testFeatures, testLabels } = getMnistData();


async function loadModel() {
    try {
        const model = await tf.loadLayersModel(`file://${modelDir}/model.json`);
        console.log('Model loaded successfully.');
        // model has to be compiled
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        return model;
    } catch (error) {
        console.error('Error loading the model:', error);
    }
}

async function run() {
    const model = await loadModel();
    if (model) {
        // Evaluate the model on the test data
        const result = model.evaluate(testFeatures, testLabels);
        const accuracy = result[1].dataSync()[0];

        console.log('Accuracy:', accuracy);
    }
}

run();

