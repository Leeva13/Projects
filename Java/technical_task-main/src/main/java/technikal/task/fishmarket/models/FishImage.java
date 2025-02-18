package technikal.task.fishmarket.models;

import jakarta.persistence.*;

@Entity
@Table(name = "fish_image")
public class FishImage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String imageFileName;

    @ManyToOne
    @JoinColumn(name = "fish_id")
    private Fish fish;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getImageFileName() {
        return imageFileName;
    }

    public void setImageFileName(String imageFileName) {
        this.imageFileName = imageFileName;
    }

    public Fish getFish() {
        return fish;
    }

    public void setFish(Fish fish) {
        this.fish = fish;
    }
}