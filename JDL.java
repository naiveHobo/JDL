import NumJ.*;

interface Layer {
	public float[][] forward(float[][] x);
	public float[][] backward(float[][] d);
	public String getName();
	public String getType();
}

class Activation implements Layer {
	public String name;
	public String type;
	private String function;
	private float[][] output;

	public Activation(String name, String function) {
		this.name = name;
		this.type = "Activation";
		this.function = function;
	}

	public float[][] forward(float[][] x) {
		if(function.equals("relu"))
			output = relu(x);
		else if(function.equals("tanh"))
			output = tanh(x);
		else if(function.equals("sigmoid"))
			output = sigmoid(x);
		else if(function.equals("softmax"))
			output = softmax(x);
		return output;
	}

	public float[][] backward(float[][] d) {
		float[][] gradient = new float[d.length][d[0].length];
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
		else if(function.equals("softmax")){
			int batch = d.length;
			try{
				gradient = NumJ.div(NumJ.sub(output, d), batch);
			}catch(Exception e){
				System.out.println(e);
			}
		}
		return gradient;
	}

	public static float[][] softmax(float[][] mat) {
		float[][] exp = NumJ.exp(mat);
		float[] rs = NumJ.sum(exp, 1);
		try{
			exp = NumJ.div(exp, rs);
		} catch(Exception e){
			System.out.println(e);
		}
		return exp;
	}

	public static float[][] sigmoid(float[][] mat) {
		float[][] temp = mat;
		try{
			temp = NumJ.reciprocal(NumJ.add(NumJ.exp(NumJ.neg(mat)), 1));	
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public static float[][] tanh(float[][] mat) {
		float[][] temp = mat;
		try{
			temp = NumJ.sub(NumJ.mul(sigmoid(NumJ.mul(mat, 2)), 2), 1);
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public static float[][] relu(float[][] mat) {
		float[][] relu = new float[mat.length][mat[0].length];
		for(int i=0;i<mat.length;i++)
			for(int j=0;j<mat[i].length;j++)
				relu[i][j] = mat[i][j]>0 ? mat[i][j] : 0;
		return relu;
	}

	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}
}

class Dense implements Layer {
	private float[][] weights;
	private float[] bias;
	private float[][] dw;
	private float[] db;
	private float[][] input;
	private String name;
	private String type;

	public Dense(String name, int input_size, int output_size) {
		this.weights = NumJ.mul(NumJ.rand(input_size, output_size), 0.01f);
		this.bias = NumJ.zeros(output_size);
		this.dw = new float[input_size][output_size];
		this.db = new float[output_size];
		this.name = name;
		this.type = "Dense";
	}

	public float[][] forward(float[][] x) {
		input = x;
		float[][] temp = x;
		try{
			temp = NumJ.add(NumJ.dot(input, weights), bias);
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public float[][] backward(float[][] d) {
		float[][] temp = d;
		try{
			dw = NumJ.dot(NumJ.transpose(input), d);
			db = NumJ.sum(d, 0);
			temp = NumJ.dot(d, NumJ.transpose(weights));
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}
}

class NeuralNetwork {
	private float loss;
	private Layer[] layers;

	public float[][] predict(float[][] x) {
		float[][] output = x;
		for(Layer l: layers){
			NumJ.display(output);
			output = l.forward(output);
		}
		return output;
	}

	public float meanSquaredError(float[][] label, float[][] pred) {
		try{
			float[][] dif = NumJ.sub(label, pred);
			return (0.5f * NumJ.sum(NumJ.mul(dif, dif)));
		}catch(Exception e){
			System.out.println(e);
		}
		return 0;
	}

	public void addLayer(Layer l) {
		if(layers==null){
			layers = new Layer[1];
			layers[0] = l;
			return;
		}
		Layer[] temp = new Layer[layers.length + 1];
		for(int i=0;i<layers.length;i++)
			temp[i] = layers[i];
		temp[temp.length-1] = l;
		layers = temp;
	}

	public Layer getLayer(String name) {
		for(Layer l: layers){
			if(name.equals(l.getName()))
				return l;
		}
		return null;
	}
}

public class JDL {
	public static void main(String args[]) {
		NeuralNetwork nn = new NeuralNetwork();
		nn.addLayer(new Dense("Dense1", 3, 4));
		nn.addLayer(new Dense("Dense2", 4, 4));
		nn.addLayer(new Activation("output", "softmax"));
		float[][] out = {{1, 2, 3}};
		float[][] l = {{0, 0, 1, 0}};
		out = nn.predict(out);
		NumJ.display(out);
		System.out.println("Loss: " + nn.meanSquaredError(l, out));
	}
}