const userName: string = 'John Doe';
let userAge: number = 25;
const isAdmin: boolean = false;

function greetUser(name: string): string {
    return `Hello, ${name}!`;
}

function calculateAge(birthYear: number): number {
    const currentYear: number = new Date().getFullYear();
    return currentYear - birthYear;
}

function isEven(num: number): boolean {
    return num % 2 === 0;
}

function displayUserInfo(name: string, age: number, admin: boolean): void {
    console.log(`Name: ${name}`);
    console.log(`Age: ${age}`);
    console.log(`Admin: ${admin ? 'Yes' : 'No'}`);
}

const numbers: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

function filterEvenNumbers(arr: number[]): number[] {
    return arr.filter(isEven);
}

const fruits: string[] = ['apple', 'banana', 'cherry', 'date', 'fig'];

function isFavoriteFruit(fruit: string): boolean {
    return fruit === 'banana';
}

const favoriteFruit: string = fruits.find(isFavoriteFruit) || 'not found';

function listFruits(fruitList: string[]): void {
    fruitList.forEach((fruit: string, index: number) => {
        console.log(`${index + 1}. ${fruit}`);
    });
}

console.log(greetUser(userName));
console.log(`User's age is: ${calculateAge(1997)}`);
displayUserInfo(userName, userAge, isAdmin);

const evenNumbers: number[] = filterEvenNumbers(numbers);
console.log('Even numbers:', evenNumbers);

listFruits(fruits);
console.log('Favorite fruit:', favoriteFruit);

let counter: number = 0;

function incrementCounter(): void {
    counter++;
}

function decrementCounter(): void {
    counter--;
}

incrementCounter();
incrementCounter();
decrementCounter();
console.log('Counter:', counter);

function add(a: number, b: number): number {
    return a + b;
}

function subtract(a: number, b: number): number {
    return a - b;
}

function multiply(a: number, b: number): number {
    return a * b;
}

function divide(a: number, b: number): number {
    if (b === 0) {
        console.log('Cannot divide by zero');
        return 0;
    }
    return a / b;
}

console.log('Add:', add(10, 5));
console.log('Subtract:', subtract(10, 5));
console.log('Multiply:', multiply(10, 5));
console.log('Divide:', divide(10, 0));

function factorial(n: number): number {
    if (n === 0) return 1;
    return n * factorial(n - 1);
}

console.log('Factorial of 5:', factorial(5));

const user: { name: string, age: number, isAdmin: boolean } = {
    name: 'Alice',
    age: 30,
    isAdmin: true
};

function getObjectKeys(obj: object): string[] {
    return Object.keys(obj);
}

console.log('User object keys:', getObjectKeys(user));

function greetAll(names: string[]): string[] {
    return names.map((name: string) => `Hello, ${name}!`);
}

console.log('Greetings:', greetAll(['Alice', 'Bob', 'Charlie']));

function toUpperCase(str: string): string {
    return str.toUpperCase();
}

console.log('Uppercase:', toUpperCase('hello'));

for (let i: number = 0; i < 5; i++) {
    console.log(`Iteration ${i + 1}`);
}

const items: string[] = ['item1', 'item2', 'item3', 'item4', 'item5'];

items.forEach((item: string, index: number) => {
    console.log(`Processing ${item} at position ${index}`);
});

const randomNumbers: number[] = Array.from({ length: 10 }, () => Math.floor(Math.random() * 100));

randomNumbers.sort((a: number, b: number) => a - b);

console.log('Sorted numbers:', randomNumbers);

const result: number = randomNumbers.reduce((acc: number, num: number) => acc + num, 0);

console.log('Sum of numbers:', result);

let toggle: boolean = true;

function toggleSwitch(): void {
    toggle = !toggle;
    console.log('Toggle is now:', toggle);
}

toggleSwitch();
toggleSwitch();

const rectangle: { width: number, height: number, area: () => number } = {
    width: 10,
    height: 5,
    area() {
        return this.width * this.height;
    }
};

console.log('Rectangle area:', rectangle.area());

const temperatures: number[] = [20, 21, 19, 22, 18, 23, 21];

const averageTemp: number = temperatures.reduce((sum: number, temp: number) => sum + temp, 0) / temperatures.length;

console.log('Average temperature:', averageTemp);

function countdown(start: number): void {
    for (let i: number = start; i > 0; i--) {
        console.log(i);
    }
    console.log('Blast off!');
}

countdown(5);

function fibonacci(n: number): number[] {
    const sequence: number[] = [0, 1];
    for (let i: number = 2; i <= n; i++) {
        sequence[i] = sequence[i - 1] + sequence[i - 2];
    }
    return sequence;
}

console.log('Fibonacci sequence up to 10:', fibonacci(10));

const animals: string[] = ['cat', 'dog', 'rabbit', 'elephant'];

const upperAnimals: string[] = animals.map((animal: string) => animal.toUpperCase());

console.log('Uppercased animals:', upperAnimals);

const shapes: string[] = ['circle', 'square', 'triangle', 'rectangle'];

shapes.forEach((shape: string) => {
    console.log(`Drawing a ${shape}`);
});

const doubleNumbers: number[] = numbers.map((num: number) => num * 2);

console.log('Doubled numbers:', doubleNumbers);

const filteredFruits: string[] = fruits.filter((fruit: string) => fruit.length > 5);

console.log('Fruits with more than 5 letters:', filteredFruits);

const isOdd: (num: number) => boolean = num => num % 2 !== 0;

const oddNumbers: number[] = numbers.filter(isOdd);

console.log('Odd numbers:', oddNumbers);
