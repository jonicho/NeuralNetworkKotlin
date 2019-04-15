package de.jrk.neuralnetwork

class NeuralNetwork(vararg neurons: Int, activationFunction: ActivationFunction = Sigmoid()) {
    val weights: Array<Matrix> = Array(neurons.size - 1) { i -> Matrix(neurons[i + 1], neurons[i]) }
    val biases: Array<Matrix> = Array(neurons.size - 1) { i -> Matrix(neurons[i + 1], 1) }
    val netInputs: Array<Matrix> = Array(neurons.size - 1) { Matrix(0, 0) }
    val activations: Array<Matrix> = Array(neurons.size - 1) { Matrix(0, 0) }
    val activationFunctions: Array<ActivationFunction> = Array(neurons.size - 1) { activationFunction }

    fun randomize(range: Double) {
        for (i in 0 until weights.size) {
            weights[i] = weights[i].map { Math.random() * 2 * range - range }
            biases[i] = biases[i].map { Math.random() * 2 * range - range }
        }
    }

    fun feedforward(inputs: Matrix): Matrix {
        for (i in 0 until activations.size) {
            netInputs[i] = (weights[i] x if (i == 0) inputs else activations[i - 1]) + biases[i]
            activations[i] = netInputs[i].map { x -> activationFunctions[i].f(x) }
        }
        return activations.last()
    }

    fun copy(): NeuralNetwork {
        val nn = NeuralNetwork(*IntArray(weights.size + 1))
        weights.forEachIndexed { i, m -> nn.weights[i] = m }
        biases.forEachIndexed { i, m -> nn.biases[i] = m }
        activations.forEachIndexed { i, m -> nn.activations[i] = m }
        netInputs.forEachIndexed { i, m -> nn.netInputs[i] = m }
        activationFunctions.forEachIndexed { i, a -> nn.activationFunctions[i] = a }
        return nn
    }
}
