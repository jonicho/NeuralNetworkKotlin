package de.jrk.neuralnetwork

class Matrix(rows: Int, cols: Int) {
    private val data: Array<DoubleArray> = Array(rows) { DoubleArray(cols) }
    val rows get() = data.size
    val cols get() = data[0].size

    constructor(rows: Int, cols: Int, initFun: (i: Int, j: Int) -> Double) : this(rows, cols) {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                data[i][j] = initFun(i, j)
            }
        }
    }

    constructor(rows: Int, cols: Int, vararg elements: Double) : this(rows, cols) {
        require(rows * cols == elements.size) { "the number of elements has to match the size of the matrix (rows * cols)!" }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                data[i][j] = elements[j * rows + i]
            }
        }
    }

    fun mapIndexed(mapFun: (x: Double, i: Int, j: Int) -> Double): Matrix {
        val result = Matrix(rows, cols)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = mapFun(this[i, j], i, j)
            }
        }
        return result
    }

    fun map(mapFun: (x: Double) -> Double) = mapIndexed { x, _, _ -> mapFun(x) }

    operator fun get(i: Int, j: Int) = data[i][j]

    operator fun plus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols) { "the size of the given matrix has to match the size of this matrix!" }
        return mapIndexed { x, i, j -> x + other[i, j] }
    }

    operator fun plus(other: Double) = map { it + other }

    operator fun unaryMinus() = map { -it }

    operator fun minus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols) { "the size of the given matrix has to match the size of this matrix!" }
        return this + -other
    }

    operator fun minus(other: Double) = map { it - other }

    operator fun times(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols) { "the size of the given matrix has to match the size of this matrix!" }
        return mapIndexed { x, i, j -> x * other[i, j] }
    }

    operator fun times(other: Double) = map { it * other }

    infix fun x(other: Matrix): Matrix {
        require(cols == other.rows) { "the rows of the given matrix must match the columns of this matrix!" }
        return Matrix(rows, other.cols) { i, j ->
            var sum = 0.0
            for (k in 0 until cols) {
                sum += this[i, k] * other[k, j]
            }
            sum
        }
    }

    fun transposed() = Matrix(cols, rows) { i, j -> this[j, i] }

    override fun toString() = data.contentDeepToString()

    override operator fun equals(other: Any?) = other is Matrix && data.contentDeepEquals(other.data)

    override fun hashCode() = data.contentDeepHashCode()
}
