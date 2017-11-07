package Dataset;

import java.util.concurrent.ThreadLocalRandom;
import java.io.*;
import java.util.*;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Image;
import javax.imageio.ImageIO;

public class MNIST {

	public static void loadTrainingData(float[][] train_x, float[][] train_y, int num) {
		File folder = new File("Dataset/training");
		BufferedImage img;
		int pos = 0;
		for (final File fileEntry : folder.listFiles()) {
			int cnt = 0;
			for (final File image_file : fileEntry.listFiles()) {
				try{
					img = ImageIO.read(MNIST.class.getResource(folder.getName() + "/" + fileEntry.getName() + "/" + image_file.getName()));
					train_x[pos] = imageToMatrix(img);
					train_y[pos][(int) (fileEntry.getName().charAt(0) - '0')] = 1.0f;
					pos++;
					cnt++;
				} catch(Exception e){
					System.out.println(e);
				}
				if(cnt==num)
					break;
			}
		}
	}

	public static void loadTestData(float[][] test_x, float[][] test_y, int num) {
		File folder = new File("Dataset/testing");
		BufferedImage img;
		int pos = 0;
		for (final File fileEntry : folder.listFiles()) {
			int cnt = 0;
			for (final File image_file : fileEntry.listFiles()) {
				try{
					img = ImageIO.read(MNIST.class.getResource(folder.getName() + "/" + fileEntry.getName() + "/" + image_file.getName()));
					test_x[pos] = imageToMatrix(img);
					test_y[pos][(int) (fileEntry.getName().charAt(0) - '0')] = 1.0f;
					pos++;
					cnt++;
				} catch(Exception e){
					System.out.println(e);
				}
				if(cnt==num)
					break;
			}
		}
	}

	public static String randomImage() {
		File folder = new File("Dataset/testing");
		int cnt = 0;
		for (final File fileEntry : folder.listFiles()) {
			for (final File image_file : fileEntry.listFiles())
				cnt++;
		}
		int randomNum = ThreadLocalRandom.current().nextInt(0, cnt);
		cnt = 0;
		for (final File fileEntry : folder.listFiles()) {
			for (final File image_file : fileEntry.listFiles()){
				if(cnt==randomNum)
					return ("Dataset/" + folder.getName() + "/" + fileEntry.getName() + "/" + image_file.getName());
				cnt++;
			}
		}
		return "Dataset/testing/0/3.png";
	}

	public static float[] imageToMatrix(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		float[] result = new float[height*width];
		int cnt = 0;

		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				int pix = image.getRGB(col, row);
				result[cnt++] = (pix>> 8) & 0x000000FF;
			}
		}

		return result;
	}

	public static BufferedImage resize(BufferedImage img, int newW, int newH) { 
		Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
		BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

		Graphics2D g2d = dimg.createGraphics();
		g2d.drawImage(tmp, 0, 0, null);
		g2d.dispose();

		return dimg;
	}
}