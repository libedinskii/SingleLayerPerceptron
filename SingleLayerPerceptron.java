import java.util.Arrays;
import java.util.Random;

public class SingleLayerPerceptron {
    private double[][] weights; // Веса для входов
    private double learningRate; // Скорость обучения
    private int numInputs; // Количество входов
    private int numOutputs; // Количество выходов

    public SingleLayerPerceptron(int numInputs, int numOutputs, double learningRate) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;
        this.weights = new double[numOutputs][numInputs]; // Инициализация весов

        // Инициализация весов случайными значениями
        Random rand = new Random();
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = rand.nextDouble() * 0.2 - 0.1; // Значения в диапазоне [-0.1, 0.1]
            }
        }
    }

    // Функция активации (пороговая функция)
    private int activation(double[] output) {
        double max = output[0];
        int maxIndex = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > max) {
                max = output[i];
                maxIndex = i;
            }
        }
        return maxIndex; // Возвращаем индекс с максимальным значением
    }

    // Прямое распространение сигнала
    public int predict(double[] input) {
        double[] output = new double[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            output[i] = 0;
            for (int j = 0; j < numInputs; j++) {
                output[i] += weights[i][j] * input[j];
            }
        }
        return activation(output);
    }

    // Обучение перцептрона
    public void train(double[][] input, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < input.length; i++) {
                int predicted = predict(input[i]);
                int target = labels[i];

                // Обновление весов
                for (int j = 0; j < numOutputs; j++) {
                    if (j == target) {
                        weights[j] = updateWeights(weights[j], input[i], learningRate);
                    } else {
                        weights[j] = updateWeights(weights[j], input[i], -learningRate);
                    }
                }
            }
        }
    }

    // Обновление весов
    private double[] updateWeights(double[] weights, double[] input, double adjustment) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += adjustment * input[i];
        }
        return weights;
    }

    public static void main(String[] args) {
        int numInputs = 100; // 10x10 = 100 входов
        int numOutputs = 10; // 10 выходов для цифр от 0 до 9
        double learningRate = 0.1;
        int epochs = 1000;

        SingleLayerPerceptron perceptron = new SingleLayerPerceptron(numInputs, numOutputs, learningRate);

        // Пример данных для тренировки
        // inputData - массив входных данных (
        double[][] inputData = {
            {0, 0, 1, 1, 1, 0, 0, 0, 0, 0,  // 0
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 1, 1, 1, 0, 0, 0, 0, 0}, // конец 0

            {0, 0, 1, 1, 1, 0, 0, 0, 0, 0,  // 1
             0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 1, 1, 1, 0, 0, 0, 0, 0}, // конец 1

            // Добавьте остальные векторы для цифр 2-9 аналогичным образом
        };

        // Массив меток для каждой цифры: 0 - 0, 1 - 1 и т.д.
        int[] labels = {0, 1 /*, ... добавьте метки для остальных данных ... */};

        // Обучение перцептрона
        perceptron.train(inputData, labels, epochs);

        // Тестирование перцептрона
        double[] testSample = { /* Ввод тестового вектора, например, для цифры 0 или 1 */ };
        int result = perceptron.predict(testSample);
        System.out.println("Результат предсказания: " + result);
    }
}   
