// Частина 1: Типи та інтерфейси
// Enum для класів персонажів
var CharacterClass;
(function (CharacterClass) {
    CharacterClass["Knight"] = "KNIGHT";
    CharacterClass["Sorcerer"] = "SORCERER";
    CharacterClass["Ranger"] = "RANGER";
})(CharacterClass || (CharacterClass = {}));
// Enum для стилів атак
var AttackStyle;
(function (AttackStyle) {
    AttackStyle["Melee"] = "MELEE";
    AttackStyle["Spell"] = "SPELL";
    AttackStyle["Bow"] = "BOW";
})(AttackStyle || (AttackStyle = {}));
// Частина 2: Функції
// 1. Функція створення нового персонажа
function createCharacter(name, characterClass) {
    var _a;
    var baseAttributes = (_a = {},
        _a[CharacterClass.Knight] = { health: 150, attackPower: 20, armor: 15, agility: 5 },
        _a[CharacterClass.Sorcerer] = { health: 100, attackPower: 30, armor: 5, agility: 10 },
        _a[CharacterClass.Ranger] = { health: 120, attackPower: 25, armor: 10, agility: 15 },
        _a);
    return {
        id: Math.floor(Math.random() * 1000),
        name: name,
        characterClass: characterClass,
        attackStyle: characterClass === CharacterClass.Knight ? AttackStyle.Melee :
            characterClass === CharacterClass.Sorcerer ? AttackStyle.Spell : AttackStyle.Bow,
        attributes: baseAttributes[characterClass],
        alive: true
    };
}
// 2. Функція розрахунку пошкодження
function calculateAttackDamage(attacker, defender) {
    var baseDamage = attacker.attributes.attackPower - defender.attributes.armor / 2;
    var critical = Math.random() < 0.2; // 20% шанс критичного удару
    var totalDamage = critical ? baseDamage * 2 : baseDamage;
    var remainingHealth = Math.max(0, defender.attributes.health - totalDamage);
    defender.attributes.health = remainingHealth;
    return {
        totalDamage: Math.round(totalDamage),
        criticalHit: critical,
        remainingHealth: Math.round(remainingHealth)
    };
}
// 3. Генерік функція для пошуку персонажа за властивістю
function findCharacterByProperty(characters, property, value) {
    return characters.find(function (char) { return char[property] === value; });
}
// 4. Функція проведення бою
function duelRound(fighter1, fighter2) {
    var damageToFighter2 = calculateAttackDamage(fighter1, fighter2);
    var damageToFighter1 = calculateAttackDamage(fighter2, fighter1);
    var result = "\n        ".concat(fighter1.name, " \u0437\u0430\u0432\u0434\u0430\u0454 \u0443\u0434\u0430\u0440\u0443 ").concat(fighter2.name, ": ").concat(damageToFighter2.totalDamage, " (\u041A\u0440\u0438\u0442: ").concat(damageToFighter2.criticalHit, ")\n        ").concat(fighter2.name, " \u0437\u0430\u0432\u0434\u0430\u0454 \u0443\u0434\u0430\u0440\u0443 ").concat(fighter1.name, ": ").concat(damageToFighter1.totalDamage, " (\u041A\u0440\u0438\u0442: ").concat(damageToFighter1.criticalHit, ")\n        \u0417\u0430\u043B\u0438\u0448\u043E\u043A \u0437\u0434\u043E\u0440\u043E\u0432'\u044F: ").concat(fighter1.name, " - ").concat(fighter1.attributes.health, ", ").concat(fighter2.name, " - ").concat(fighter2.attributes.health, "\n    ");
    // Перевірка на виживання
    if (fighter1.attributes.health <= 0)
        fighter1.alive = false;
    if (fighter2.attributes.health <= 0)
        fighter2.alive = false;
    return result;
}
// Частина 3: Практичне застосування
// Створюємо масив персонажів
var characters = [
    createCharacter("Артур", CharacterClass.Knight),
    createCharacter("Моргана", CharacterClass.Sorcerer),
    createCharacter("Леголас", CharacterClass.Ranger)
];
// Показуємо створених персонажів
console.log("Створені персонажі:", characters);
// Знаходимо персонажа за властивістю
var foundCharacter = findCharacterByProperty(characters, "characterClass", CharacterClass.Knight);
console.log("Знайдений персонаж:", foundCharacter);
// Проводимо раунд бою
var fightResult = duelRound(characters[0], characters[1]);
console.log(fightResult);
// Статистика після бою
console.log("Стан персонажів після бою:", characters);
