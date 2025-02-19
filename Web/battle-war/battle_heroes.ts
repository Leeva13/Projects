// Enum для класів персонажів
enum CharacterClass {
    Knight = "KNIGHT",
    Wizard = "WIZARD",
    Ranger = "RANGER"
}

// Enum для типів шкоди
enum DamageType {
    Melee = "MELEE",
    Spell = "SPELL",
    Bow = "BOW"
}

// Інтерфейс для характеристик персонажа
interface CharacterStats {
    health: number;
    attack: number;
    defense: number;
    speed: number;
    energy: number;
}

// Інтерфейс для персонажа
interface Character {
    id: number;
    nickname: string;
    characterClass: CharacterClass;
    damageType: DamageType;
    stats: CharacterStats;
    alive: boolean;
}

// Тип результату атаки
type DamageResult = {
    inflictedDamage: number;
    wasCritical: boolean;
    remainingHealth: number;
}

// Лічильник для ID персонажів
let characterId = 1000;

/**
 * Функція для створення персонажа
 * @param nickname Ім'я персонажа
 * @param charClass Клас персонажа (Knight, Wizard, Ranger)
 * @returns Новий об'єкт персонажа
 */
function createCharacter(nickname: string, charClass: CharacterClass): Character {
    let initialStats: CharacterStats;
    switch (charClass) {
        case CharacterClass.Knight:
            initialStats = { health: 150, attack: 18, defense: 20, speed: 8, energy: 100 };
            break;
        case CharacterClass.Wizard:
            initialStats = { health: 90, attack: 35, defense: 5, speed: 18, energy: 150 };
            break;
        case CharacterClass.Ranger:
            initialStats = { health: 110, attack: 25, defense: 10, speed: 14, energy: 50 };
            break;
    }

    return {
        id: characterId++,
        nickname,
        characterClass: charClass,
        damageType: determineDamageType(charClass),
        stats: initialStats,
        alive: true
    };
}

/**
 * Визначає тип атаки на основі класу персонажа
 */
function determineDamageType(charClass: CharacterClass): DamageType {
    switch (charClass) {
        case CharacterClass.Knight: return DamageType.Melee;
        case CharacterClass.Wizard: return DamageType.Spell;
        case CharacterClass.Ranger: return DamageType.Bow;
    }
}

/**
 * Обчислення пошкодження між атакуючим та захисником
 */
function computeDamage(attacker: Character, defender: Character): DamageResult {
    const baseDamage = attacker.stats.attack - defender.stats.defense / 2;
    const finalDamage = baseDamage > 0 ? Math.floor(baseDamage) : 1;

    const wasCritical = Math.random() < 0.25; // 25% шанс критичного удару
    const criticalDamage = wasCritical ? finalDamage * 1.5 : finalDamage;

    defender.stats.health -= criticalDamage;
    if (defender.stats.health <= 0) {
        defender.alive = false;
        defender.stats.health = 0;
    }

    return {
        inflictedDamage: Math.floor(criticalDamage),
        wasCritical,
        remainingHealth: defender.stats.health
    };
}

/**
 * Генерік функція для пошуку персонажа за властивістю
 */
function searchCharacter<T extends keyof Character>(
    characters: Character[],
    property: T,
    value: Character[T]
): Character | undefined {
    return characters.find(char => char[property] === value);
}

/**
 * Функція для бою між двома персонажами
 */
function fightRound(char1: Character, char2: Character): string {
    if (!char1.alive || !char2.alive) {
        return "Один із персонажів вже мертвий!";
    }

    const firstAttacker = char1.stats.speed >= char2.stats.speed ? char1 : char2;
    const secondAttacker = firstAttacker === char1 ? char2 : char1;

    const attack1 = computeDamage(firstAttacker, secondAttacker);
    let result = `${firstAttacker.nickname} атакує ${secondAttacker.nickname} і завдає ${attack1.inflictedDamage} ${attack1.wasCritical ? "(Критичний удар!)" : ""}. Залишилось HP: ${secondAttacker.stats.health}.\n`;

    if (!secondAttacker.alive) {
        result += `${secondAttacker.nickname} програв бій!`;
        return result;
    }

    const attack2 = computeDamage(secondAttacker, firstAttacker);
    result += `${secondAttacker.nickname} відповідає та завдає ${attack2.inflictedDamage} ${attack2.wasCritical ? "(Критичний удар!)" : ""}. Залишилось HP: ${firstAttacker.stats.health}.\n`;

    if (!firstAttacker.alive) {
        result += `${firstAttacker.nickname} програв бій!`;
    }

    return result;
}

// === Демонстрація роботи ===
const characters: Character[] = [
    createCharacter("Артур", CharacterClass.Knight),
    createCharacter("Гендальф", CharacterClass.Wizard),
    createCharacter("Робін", CharacterClass.Ranger)
];

// Пошук персонажа
const searchedCharacter = searchCharacter(characters, "characterClass", CharacterClass.Wizard);
console.log("Знайдений персонаж:", searchedCharacter);

// Бій між персонажами
console.log(fightRound(characters[0], characters[1]));
console.log(fightRound(characters[0], characters[2]));
