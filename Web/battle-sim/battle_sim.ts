// Частина 1: Типи та інтерфейси

// Enum для класів персонажів
enum CharacterClass {
    Knight = "KNIGHT",
    Sorcerer = "SORCERER",
    Ranger = "RANGER"
}

// Enum для стилів атак
enum AttackStyle {
    Melee = "MELEE",
    Spell = "SPELL",
    Bow = "BOW"
}

// Інтерфейс для характеристик персонажа
interface CharacterAttributes {
    health: number;
    attackPower: number;
    armor: number;
    agility: number;
}

// Інтерфейс для персонажа
interface Character {
    id: number;
    name: string;
    characterClass: CharacterClass;
    attackStyle: AttackStyle;
    attributes: CharacterAttributes;
    alive: boolean;
}

// Тип для результату атаки
type DamageResult = {
    totalDamage: number;
    criticalHit: boolean;
    remainingHealth: number;
}


// Частина 2: Функції

// 1. Функція створення нового персонажа
function createCharacter(name: string, characterClass: CharacterClass): Character {
    const baseAttributes: Record<CharacterClass, CharacterAttributes> = {
        [CharacterClass.Knight]: { health: 150, attackPower: 20, armor: 15, agility: 5 },
        [CharacterClass.Sorcerer]: { health: 100, attackPower: 30, armor: 5, agility: 10 },
        [CharacterClass.Ranger]: { health: 120, attackPower: 25, armor: 10, agility: 15 }
    };

    return {
        id: Math.floor(Math.random() * 1000),
        name,
        characterClass,
        attackStyle:
            characterClass === CharacterClass.Knight ? AttackStyle.Melee :
            characterClass === CharacterClass.Sorcerer ? AttackStyle.Spell : AttackStyle.Bow,
        attributes: baseAttributes[characterClass],
        alive: true
    };
}

// 2. Функція розрахунку пошкодження
function calculateAttackDamage(attacker: Character, defender: Character): DamageResult {
    const baseDamage = attacker.attributes.attackPower - defender.attributes.armor / 2;
    const critical = Math.random() < 0.2; // 20% шанс критичного удару
    const totalDamage = critical ? baseDamage * 2 : baseDamage;

    const remainingHealth = Math.max(0, defender.attributes.health - totalDamage);
    defender.attributes.health = remainingHealth;

    return {
        totalDamage: Math.round(totalDamage),
        criticalHit: critical,
        remainingHealth: Math.round(remainingHealth)
    };
}

// 3. Генерік функція для пошуку персонажа за властивістю
function findCharacterByProperty<T extends keyof Character>(
    characters: Character[],
    property: T,
    value: Character[T]
): Character | undefined {
    return characters.find((char) => char[property] === value);
}

// 4. Функція проведення бою
function duelRound(fighter1: Character, fighter2: Character): string {
    const damageToFighter2 = calculateAttackDamage(fighter1, fighter2);
    const damageToFighter1 = calculateAttackDamage(fighter2, fighter1);

    const result = `
        ${fighter1.name} завдає удару ${fighter2.name}: ${damageToFighter2.totalDamage} (Крит: ${damageToFighter2.criticalHit})
        ${fighter2.name} завдає удару ${fighter1.name}: ${damageToFighter1.totalDamage} (Крит: ${damageToFighter1.criticalHit})
        Залишок здоров'я: ${fighter1.name} - ${fighter1.attributes.health}, ${fighter2.name} - ${fighter2.attributes.health}
    `;

    // Перевірка на виживання
    if (fighter1.attributes.health <= 0) fighter1.alive = false;
    if (fighter2.attributes.health <= 0) fighter2.alive = false;

    return result;
}

// Частина 3: Практичне застосування

// Створюємо масив персонажів
const characters: Character[] = [
    createCharacter("Артур", CharacterClass.Knight),
    createCharacter("Моргана", CharacterClass.Sorcerer),
    createCharacter("Леголас", CharacterClass.Ranger)
];

// Показуємо створених персонажів
console.log("Створені персонажі:", characters);

// Знаходимо персонажа за властивістю
const foundCharacter = findCharacterByProperty(characters, "characterClass", CharacterClass.Knight);
console.log("Знайдений персонаж:", foundCharacter);

// Проводимо раунд бою
const fightResult = duelRound(characters[0], characters[1]);
console.log(fightResult);

// Статистика після бою
console.log("Стан персонажів після бою:", characters);
