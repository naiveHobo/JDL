import java.io.*;
import java.util.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class dataload {

	public static void main(String args[]) throws IOException {
		BufferedImage hugeImage = ImageIO.read(dataload.class.getResource("mnist/training/0/1.png"));
		int[][] res = imageToMatrix(hugeImage);
		File folder = new File("mnist/training");
		listFilesForFolder(folder);
	}

	public static int[][] imageToMatrix(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		int[][] result = new int[height][width];

		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				int pix = image.getRGB(col, row);
				result[row][col] = (pix>> 16) & 0x000000FF;
			}
		}

		return result;
	}

	public static void listFilesForFolder(final File folder) {
    	for (final File fileEntry : folder.listFiles()) {
        	if (fileEntry.isDirectory()) {
            	listFilesForFolder(fileEntry);
			} else {
				System.out.println(folder.getName() + "/" + fileEntry.getName());
			}
		}
	}
}