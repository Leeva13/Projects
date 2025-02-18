package technikal.task.fishmarket.controllers;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import jakarta.validation.Valid;
import technikal.task.fishmarket.models.Fish;
import technikal.task.fishmarket.models.FishDto;
import technikal.task.fishmarket.models.FishImage;
import technikal.task.fishmarket.services.FishRepository;

@Controller
@RequestMapping("/fish")
public class FishController {

	@Autowired
	private FishRepository fishRepository;

	@GetMapping({"", "/"})
	public String showFishList(Model model) {
		List<Fish> fishlist = fishRepository.findAll(Sort.by(Sort.Direction.DESC, "id"));
		model.addAttribute("fishlist", fishlist);
		return "index";
	}

	@GetMapping("/create")
	public String showCreateForm(Model model) {
		model.addAttribute("fishDto", new FishDto());
		return "createFish";
	}

	@GetMapping("/delete")
	public String deleteFish(@RequestParam Long id) {
		try {
			Fish fish = fishRepository.findById(id).orElse(null);
			if (fish != null) {
				// Видалення усіх фото рибки
				for (FishImage image : fish.getImages()) {
					Path imagePath = Paths.get("uploads/images/" + image.getImageFileName());
					Files.deleteIfExists(imagePath);
				}
				fishRepository.delete(fish);
				System.out.println("Рибка з ID " + id + " видалена.");
			} else {
				System.out.println("Рибка з ID " + id + " не знайдена.");
			}
		} catch (Exception ex) {
			System.out.println("Exception: " + ex.getMessage());
		}
		return "redirect:/fish";
	}

	@PostMapping("/create")
	public String createFish(@Valid @ModelAttribute FishDto fishDto, BindingResult result) {
		if (result.hasErrors()) {
			return "createFish";
		}

		Fish fish = new Fish();
		fish.setName(fishDto.getName());
		fish.setPrice(fishDto.getPrice());
		fish.setCatchDate(new Date());

		List<FishImage> images = new ArrayList<>();
		for (MultipartFile file : fishDto.getImageFiles()) {
			if (!file.isEmpty()) {
				try {
					String contentType = file.getContentType();
					if (contentType == null || !contentType.startsWith("image/")) {
						result.addError(new FieldError("fishDto", "imageFiles", "Файл повинен бути зображенням"));
						continue;
					}

					// Створюємо папку для зображень, якщо вона не існує
					Path uploadDir = Paths.get("uploads/images/");
					if (!Files.exists(uploadDir)) {
						Files.createDirectories(uploadDir);
						System.out.println("Створено папку: " + uploadDir.toAbsolutePath());
					}

					// Унікалізуємо ім'я файлу, щоб уникнути конфліктів
					String fileName = System.currentTimeMillis() + "_" + file.getOriginalFilename();
					Path path = uploadDir.resolve(fileName);
					try (InputStream inputStream = file.getInputStream()) {
						Files.copy(inputStream, path, StandardCopyOption.REPLACE_EXISTING);
						System.out.println("Збережено файл: " + path.toAbsolutePath());
					}

					// Додаємо інформацію про файл до бази даних
					FishImage fishImage = new FishImage();
					fishImage.setImageFileName(fileName);
					fishImage.setFish(fish);
					images.add(fishImage);
				} catch (IOException e) {
					System.err.println("Не вдалося завантажити файл: " + e.getMessage());
					result.addError(new FieldError("fishDto", "imageFiles", "Не вдалося завантажити файл: " + e.getMessage()));
				}
			}
		}

		fish.setImages(images);
		fishRepository.save(fish);
		System.out.println("Рибка створена: " + fish.getName());

		return "redirect:/fish";
	}
}
