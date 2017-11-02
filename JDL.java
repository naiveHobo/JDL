import NumJ.*;

interface Layer {
	public float[][] forward(float[][] x);
	public float[][] backward(float[][] d);
}

class Activation implements Layer {
	private String function;
	private float[][] output;
	private float[][] gradient;

	public Activation(String function) {
		this.function = function;
	}

	public float[][] forward(float[][] x) {
		if(function.equals("relu"))
			output = relu(x);
		else if(function.equals("tanh"))
			output = tanh(x);
		else if(function.equals("sigmoid"))
			output = sigmoid(x);
		return output;
	}

	public float[][] backward(float[][] d) {
		if(function.equals("relu")){
			gradient = d;
			for(int i=0;i<output.length;i++){
				for(int j=0;j<output[0].length;j++){
					if(output[i][j]==0)
						gradient[i][j] = 0;
				}
			}
		}
		else if(function.equals("tanh")){
			try{
				gradient = NumJ.mul(d, NumJ.add(NumJ.neg(NumJ.mul(output, output)), 1.0f));
			}catch(Exception e){
				System.out.println(e);
			}
		}
		else if(function.equals("sigmoid")){
			try{
				gradient = NumJ.mul(d, NumJ.mul(output, NumJ.add(NumJ.neg(output), 1.0f)));
			}catch(Exception e){
				System.out.println(e);
			}
		}
		return gradient;
	}

	static float[][] sigmoid(float[][] mat1) {
		return NumJ.reciprocal(NumJ.add(NumJ.exp(NumJ.neg(mat1)), 1));
	}

	static float[][] tanh(float[][] mat1) {
		return NumJ.sub(NumJ.mul(sigmoid(NumJ.mul(mat1, 2)), 2), 1);
	}

	static float[][] relu(float[][] mat1) {
		float[][] relu = new float[mat1.length][mat1[0].length];
		for(int i=0;i<mat1.length;i++)
			for(int j=0;j<mat1[i].length;j++)
				relu[i][j] = mat1[i][j]>0 ? mat1[i][j] : 0;
		return relu;
	}
}


class NeuralNetwork {
	float[][] w1;
	float[][] b1;
	float[][] w2;
	float[][] b2;
	Layer[] layers;

	public NeuralNetwork(int input_size, int hidden_size, int output_size) {
		w1 = NumJ.mul(NumJ.rand(input_size, hidden_size), 0.01f);
		b1 = NumJ.zeros(hidden_size);
		w2 = NumJ.mul(NumJ.rand(hidden_size, output_size), 0.01f);
		b2 = NumJ.zeros(output_size);
	}

	// public float[] predict(float[][] x) {

	// }
}

public class Test {
	public static void main(String args[]) {
		float[][] mat1 = {{2.0f, 2.0f}, {2.0f, 2.0f}, {2.0f, 2.0f}};
		float[] mat2 = {2.0f};
		float[][] res;
		float sum = 0;
		try{
			res = NumJ.add(mat1, mat2);
			// res = NumJ.mul(res, 0.01f);
			// sum = NumJ.sum(res);
			// System.out.println("Std: " + sum);
			// sum /= 10000;
			// res = NumJ.sub(res, sum);
			// res = NumJ.mul(res, res);
			// sum = NumJ.sum(res);
			// sum = sum/10000;
			// sum = (float) Math.sqrt(sum);
			display(res);
			// System.out.println("Std: " + sum);
		}
		catch(Exception e){
			System.out.println(e);
		}
	}

	static void display(float[][] mat) {
		for(int i=0;i<mat.length;i++){
			for(int j=0;j<mat[0].length;j++)
				System.out.print(mat[i][j] + " ");
			System.out.println();
		}
	}
}