package neuralnetwork

import kotlin.math.absoluteValue
import kotlin.math.pow

interface ActivationFunction {
    fun f(x: Double): Double
    fun df(x: Double, fx: Double = Double.NaN): Double
    override fun toString(): String
}

class Identity : ActivationFunction {
    override fun f(x: Double) = x
    override fun df(x: Double, fx: Double) = 1.0
    override fun toString() = "identity"
}

class BinaryStep : ActivationFunction {
    override fun f(x: Double) = if (x < 0) .0 else 1.0
    override fun df(x: Double, fx: Double) = .0
    override fun toString() = "binary_step"
}

class Sigmoid : ActivationFunction {
    override fun f(x: Double) = 1 / (1 + Math.exp(-x))
    override fun df(x: Double, fx: Double) = if (fx == Double.NaN) f(x).let { it * (1 - it) } else fx * (1 - fx)
    override fun toString() = "sigmoid"
}

class TanH : ActivationFunction {
    override fun f(x: Double) = Math.tanh(x)
    override fun df(x: Double, fx: Double) = if (fx == Double.NaN) f(x).let { 1 - it.pow(2) } else 1 - fx.pow(2)
    override fun toString() = "tanh"
}

class ArcTan : ActivationFunction {
    override fun f(x: Double) = Math.atan(x)
    override fun df(x: Double, fx: Double) = 1 / (x.pow(2) + 1)
    override fun toString() = "arctan"
}

class Softsign : ActivationFunction {
    override fun f(x: Double) = x / (1 + x.absoluteValue)
    override fun df(x: Double, fx: Double) = 1 / (1 + x.absoluteValue).pow(2)
    override fun toString() = "softsign"
}

class ReLU : ActivationFunction {
    override fun f(x: Double) = if (x < 0) .0 else x
    override fun df(x: Double, fx: Double) = if (x < 0) .0 else 1.0
    override fun toString() = "relu"
}

class Sinusoid : ActivationFunction {
    override fun f(x: Double) = Math.sin(x)
    override fun df(x: Double, fx: Double) = Math.cos(x)
    override fun toString() = "sinusoid"
}

class Sinc : ActivationFunction {
    override fun f(x: Double) = if (x == .0) 1.0 else Math.sin(x) / x
    override fun df(x: Double, fx: Double) = if (x == .0) .0 else Math.cos(x) / x - if (fx == Double.NaN) Math.sin(x) / x.pow(2) else fx / x
    override fun toString() = "sinc"
}

class Gaussian : ActivationFunction {
    override fun f(x: Double) = Math.exp(-x.pow(2))
    override fun df(x: Double, fx: Double) = -2 * x * if (fx == Double.NaN) f(x) else fx
    override fun toString() = "gaussian"
}
