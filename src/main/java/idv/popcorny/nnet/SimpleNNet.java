package idv.popcorny.nnet;

import org.la4j.Matrix;
import java.util.Random;

/**
 * The code was inspired by http://iamtrask.github.io/2015/07/12/basic-python-network/
 */
public class SimpleNNet {
    public static void main(String[] args) {
        Matrix X = Matrix.from2DArray(new double[][]{
                {0, 0, 1},
                {0, 1, 1},
                {1, 0, 1},
                {1, 1, 1},
        });
        Matrix Y = Matrix.from2DArray(new double[][]{
                {0, 1, 1, 0}
        }).transpose();

        Random random = new Random();
        Matrix Syn0 = Matrix.random(3, 4, random)
                .multiply(2)
                .subtract(1);
        Matrix Syn1 = Matrix.random(4, 1, random)
                .multiply(2)
                .subtract(1);

        Matrix X0 = X;
        Matrix X1 = null;
        Matrix X2 = null;
        for (int loop=1; loop<=50000; loop++) {
            // Forward
            // s1 = x0 . syn0
            // x1 = sigmoid(s1)
            Matrix S1 = X0.multiply(Syn0);
            X1 = S1.transform((i, j, value) -> sigmoid(value));
            // s2 = x1 . syn1
            // x2 = sigmoid(x1 . syn1)
            Matrix S2 = X1.multiply(Syn1);
            X2 = S2.transform((i, j, value) -> sigmoid(value));

            // Backward
            // l2_delta = 2*(y - x2)*sigmoid'(s2)
            Matrix L2_delta = Y.subtract(X2).multiply(2).hadamardProduct(
                    S2.transform((i, j, value) -> sigmoid_deriv(value))
            );

            // l1_delta = l2_delta.dot(syn1.T) * sigmoid'(s1)
            Matrix L1_delta = L2_delta.multiply(Syn1.transpose()).hadamardProduct(
                    S1.transform((i, j, value) -> sigmoid_deriv(value))
            );

            // Update the weights
            // syn1 += x1.T.dot(l2_delta)
            Syn1 = Syn1.add(X1.transpose().multiply(L2_delta));
            // syn0 += X.T.dot(l1_delta)
            Syn0 = Syn0.add(X0.transpose().multiply(L1_delta));
        }

        System.out.println(X2);
    }

    /**
     * Sigmoid Function: https://en.wikipedia.org/wiki/Sigmoid_function
     */
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivation of Sigmoid Function: https://en.wikipedia.org/wiki/Sigmoid_function
     */
    private static double sigmoid_deriv(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }
}
