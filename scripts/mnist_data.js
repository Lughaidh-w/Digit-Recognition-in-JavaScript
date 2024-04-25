const tf = require('@tensorflow/tfjs-node');
const mnist = require('mnist');

function getMnistData() {
    const set = mnist.set(8000, 2000);
    const trainingSet = set.training;
    const testSet = set.test;

    const trainingFeatures = trainingSet.map(item => item.input);
    const trainingLabels = trainingSet.map(item => item.output);
    const testFeatures = testSet.map(item => item.input);
    const testLabels = testSet.map(item => item.output);

    const trainingFeaturesTensor = tf.tensor2d(trainingFeatures);
    const trainingLabelsTensor = tf.tensor2d(trainingLabels);
    const testFeaturesTensor = tf.tensor2d(testFeatures);
    const testLabelsTensor = tf.tensor2d(testLabels);

    return {
        trainingFeatures: trainingFeaturesTensor,
        trainingLabels: trainingLabelsTensor,
        testFeatures: testFeaturesTensor,
        testLabels: testLabelsTensor
    };
}

module.exports = getMnistData;