package com.brazhnyk.Unveil.component;

import com.brazhnyk.Unveil.model.Question;
import com.brazhnyk.Unveil.repository.QuestionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class DatabaseInitialization implements CommandLineRunner {

    @Autowired
    private QuestionRepository questionRepository;

    @Override
    public void run(String... args) throws Exception {
        if (questionRepository.count() == 0) {
            questionRepository.save(new Question(null, "Який найдавніший населений континент на Землі?", "Африка", "Азія", "Європа", "Європа"));
            questionRepository.save(new Question(null, "Яка тварина має найбільший мозок у порівнянні з розміром тіла?", "Слон", "Восьминіг", "Дельфін", "Восьминіг"));
            questionRepository.save(new Question(null, "Яка найвища гора в сонячній системі?", "Еверест (Земля)", "Мауна-Кеа (Гаваї)", "Олімп (Марс)", "Олімп (Марс)"));
            questionRepository.save(new Question(null, "Хто написав роман `Володар перснів`?", "Дж. Р. Р. Толкін", "К.С. Льюїс", "Джоан Роулінг", "Дж. Р. Р. Толкін"));
            questionRepository.save(new Question(null, "Що таке реляційна база даних?", "База даних, що зберігає дані в структурах дерева", "База даних, що використовує таблиці для зберігання даних", "База даних, яка використовує стек даних", "База даних, що використовує таблиці для зберігання даних"));
            questionRepository.save(new Question(null, "Яка найотруйніша тварина у світі?", "Кубомедуза", "Павук-сиднейський вовк", "Грайворон", "Кубомедуза"));
            questionRepository.save(new Question(null, "Яка найглибша частина океану?", "Південно-Сандвічевий жолоб", "Маріанська западина", "Жолоб Тонга", "Маріанська западина"));
            questionRepository.save(new Question(null, "Яка найшвидша тварина на Землі?", "Гепард", "Перепончатокрилий", "Сокіл-сапсан", "Перепончатокрилий"));
            questionRepository.save(new Question(null, "Яка найчисленніша тварина на Землі?", "Мурахи", "Терміти", "Комарі", "Комарі"));
            questionRepository.save(new Question(null, "Хто був першим президентом України?", "Леонід Кравчук", "Леонід Кучма", "Віктор Ющенко", "Леонід Кравчук"));
            questionRepository.save(new Question(null, "Який хімічний елемент необхідний для фотосинтезу?", "Азот", "Кисень", "Хлор", "Хлор"));
            questionRepository.save(new Question(null, "Який континент є найсухішим на Землі?", "Африка", "Антарктида", "Австралія", "Антарктида"));
            questionRepository.save(new Question(null, "Який вулкан вважається найвищим у світі?", "Охос-дель-Саладо", "Еверест", "Мауна Кеа", "Охос-дель-Саладо"));
            questionRepository.save(new Question(null, "Яка країна має найбільшу кількість островів у світі?", "Індонезія", "Філіппіни", "Японія", "Індонезія"));
            questionRepository.save(new Question(null, "Яка країна має найбільше кількість офіційних мов?", "Індія", "Швейцарія", "Канада", "Індія"));
            questionRepository.save(new Question(null, "Яка країна є найбільшою виробницею шоколаду у світі?", "Швейцарія", "Бельгія", "Нідерланди", "Нідерланди"));
            questionRepository.save(new Question(null, "Яка країна є рідною для мармеладу?", "Англія", "Франція", "Португалія", "Португалія"));
            questionRepository.save(new Question(null, "Яка країна має найбільшу кількість пірамід?", "Єгипет", "Мексика", "Судан", "Судан"));
            questionRepository.save(new Question(null, "Скільки океанів на Землі?", "4", "5", "6", "5"));
            questionRepository.save(new Question(null, "Яка країна має найбільше офіційних свят?", "Китай", "Індія", "Японія", "Індія"));
            questionRepository.save(new Question(null, "Який континент є найбільшим вибуховим?", "Африка", "Євразія", "Північна Америка", "Євразія"));
            questionRepository.save(new Question(null, "Скільки пальців на передній лапі у ведмедя?", "4", "5", "6", "5"));
            questionRepository.save(new Question(null, "Що таке API в програмуванні?", "Інтерфейс програмного забезпечення", "Відкритий доступ до інтернет-ресурсів", "Програмне забезпечення для статистичного аналізу", "Інтерфейс програмного забезпечення"));
            questionRepository.save(new Question(null, "Скільки днів у високосному році?", "366", "365", "364", "366"));
            questionRepository.save(new Question(null, "Яка кавова культура є найбільш поширеною у світі?", "Еспрессо", "Капучино", "Латте", "Еспрессо"));
            questionRepository.save(new Question(null, "Скільки основних кольорів у кубика Рубіка?", "4", "6", "8", "6"));
            questionRepository.save(new Question(null, "Яка країна виробляє найбільше бавовни у світі?", "Китай", "Індія", "США", "Індія"));
            questionRepository.save(new Question(null, "Яка мова програмування частіше за все використовується для створення веб-додатків?", "C++", "Python", "Java", "Java"));
            questionRepository.save(new Question(null, "Яка тварина має найбільше кількість зубів?", "Слоносердце", "Кит", "Африканський слон", "Слоносердце"));
            questionRepository.save(new Question(null, "Як називається процес конвертації текстового представлення даних у бінарний формат?", "Десеріалізація", "Серіалізація", "Трансформація", "Серіалізація"));
            questionRepository.save(new Question(null, "Як називається мова програмування, яка частково компілюється, а частково інтерпретується?", "Java", "C#", "Python", "Python"));
            questionRepository.save(new Question(null, "Який протокол використовується для безпечної передачі даних у мережі Інтернет?", "FTP", "SSH", "HTTP", "SSH"));
        }
    }
}
