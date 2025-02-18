package technikal.task.fishmarket.models;

import jakarta.persistence.Column;
import org.springframework.web.multipart.MultipartFile;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;

import java.util.List;

public class FishDto {

	@NotEmpty(message = "Потрібна назва рибки")
	private String name;

	@Min(value = 0, message = "Ціна не може бути від'ємною")
	private double price;

	private List<MultipartFile> imageFiles;

	@Column(name = "weight")
	private Double weight;

	public Double getWeight() {
		return weight;
	}

	public void setWeight(Double weight) {
		this.weight = weight;
	}

	public @NotEmpty(message = "потрібна назва рибки") String getName() {
		return name;
	}

	public void setName(@NotEmpty(message = "потрібна назва рибки") String name) {
		this.name = name;
	}

	@Min(0)
	public double getPrice() {
		return price;
	}

	public void setPrice(@Min(0) double price) {
		this.price = price;
	}

	public List<MultipartFile> getImageFiles() {
		return imageFiles;
	}

	public void setImageFiles(List<MultipartFile> imageFiles) {
		this.imageFiles = imageFiles;
	}
}