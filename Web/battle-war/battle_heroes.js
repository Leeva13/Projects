// Enum для класів персонажів
var CharacterClass;
(function (CharacterClass) {
    CharacterClass["Knight"] = "KNIGHT";
    CharacterClass["Wizard"] = "WIZARD";
    CharacterClass["Ranger"] = "RANGER";
})(CharacterClass || (CharacterClass = {}));
// Enum для типів шкоди
var DamageType;
(function (DamageType) {
    DamageType["Melee"] = "MELEE";
    DamageType["Spell"] = "SPELL";
    DamageType["Bow"] = "BOW";
})(DamageType || (DamageType = {}));
// Лічильник для ID персонажів
var characterId = 1000;
/**
 * Функція для створення персонажа
 * @param nickname Ім'я персонажа
 * @param charClass Клас персонажа (Knight, Wizard, Ranger)
 * @returns Новий об'єкт персонажа
 */
function createCharacter(nickname, charClass) {
    var initialStats;
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
        nickname: nickname,
        characterClass: charClass,
        damageType: determineDamageType(charClass),
        stats: initialStats,
        alive: true
    };
}
/**
 * Визначає тип атаки на основі класу персонажа
 */
function determineDamageType(charClass) {
    switch (charClass) {
        case CharacterClass.Knight: return DamageType.Melee;
        case CharacterClass.Wizard: return DamageType.Spell;
        case CharacterClass.Ranger: return DamageType.Bow;
    }
}
/**
 * Обчислення пошкодження між атакуючим та захисником
 */
function computeDamage(attacker, defender) {
    var baseDamage = attacker.stats.attack - defender.stats.defense / 2;
    var finalDamage = baseDamage > 0 ? Math.floor(baseDamage) : 1;
    var wasCritical = Math.random() < 0.25; // 25% шанс критичного удару
    var criticalDamage = wasCritical ? finalDamage * 1.5 : finalDamage;
    defender.stats.health -= criticalDamage;
    if (defender.stats.health <= 0) {
        defender.alive = false;
        defender.stats.health = 0;
    }
    return {
        inflictedDamage: Math.floor(criticalDamage),
        wasCritical: wasCritical,
        remainingHealth: defender.stats.health
    };
}
/**
 * Генерік функція для пошуку персонажа за властивістю
 */
function searchCharacter(characters, property, value) {
    return characters.find(function (char) { return char[property] === value; });
}
/**
 * Функція для бою між двома персонажами
 */
function fightRound(char1, char2) {
    if (!char1.alive || !char2.alive) {
        return "Один із персонажів вже мертвий!";
    }
    var firstAttacker = char1.stats.speed >= char2.stats.speed ? char1 : char2;
    var secondAttacker = firstAttacker === char1 ? char2 : char1;
    var attack1 = computeDamage(firstAttacker, secondAttacker);
    var result = "".concat(firstAttacker.nickname, " \u0430\u0442\u0430\u043A\u0443\u0454 ").concat(secondAttacker.nickname, " \u0456 \u0437\u0430\u0432\u0434\u0430\u0454 ").concat(attack1.inflictedDamage, " ").concat(attack1.wasCritical ? "(Критичний удар!)" : "", ". \u0417\u0430\u043B\u0438\u0448\u0438\u043B\u043E\u0441\u044C HP: ").concat(secondAttacker.stats.health, ".\n");
    if (!secondAttacker.alive) {
        result += "".concat(secondAttacker.nickname, " \u043F\u0440\u043E\u0433\u0440\u0430\u0432 \u0431\u0456\u0439!");
        return result;
    }
    var attack2 = computeDamage(secondAttacker, firstAttacker);
    result += "".concat(secondAttacker.nickname, " \u0432\u0456\u0434\u043F\u043E\u0432\u0456\u0434\u0430\u0454 \u0442\u0430 \u0437\u0430\u0432\u0434\u0430\u0454 ").concat(attack2.inflictedDamage, " ").concat(attack2.wasCritical ? "(Критичний удар!)" : "", ". \u0417\u0430\u043B\u0438\u0448\u0438\u043B\u043E\u0441\u044C HP: ").concat(firstAttacker.stats.health, ".\n");
    if (!firstAttacker.alive) {
        result += "".concat(firstAttacker.nickname, " \u043F\u0440\u043E\u0433\u0440\u0430\u0432 \u0431\u0456\u0439!");
    }
    return result;
}
// === Демонстрація роботи ===
var characters = [
    createCharacter("Артур", CharacterClass.Knight),
    createCharacter("Гендальф", CharacterClass.Wizard),
    createCharacter("Робін", CharacterClass.Ranger)
];
// Пошук персонажа
var searchedCharacter = searchCharacter(characters, "characterClass", CharacterClass.Wizard);
console.log("Знайдений персонаж:", searchedCharacter);
// Бій між персонажами
console.log(fightRound(characters[0], characters[1]));
console.log(fightRound(characters[0], characters[2]));
