// Крок 1: Визначення базових типів


// Визначаємо тип для днів тижня
type DayOfWeek = "Monday" | "Tuesday" | "Wednesday" | "Thursday" | "Friday";

// Визначаємо тип для часових слотів занять
type TimeSlot = "8:30-10:00" | "10:15-11:45" | "12:15-13:45" | "14:00-15:30" | "15:45-17:15";

// Визначаємо тип для типів занять
type CourseType = "Lecture" | "Seminar" | "Lab" | "Practice";


// Крок 2: Створення основних структур


// Визначаємо тип для професора
type Professor = {
    id: number;
    name: string;
    department: string;
};

// Визначаємо тип для аудиторії
type Classroom = {
    number: string;
    capacity: number;
    hasProjector: boolean;
};

// Визначаємо тип для курсу
type Course = {
    id: number;
    name: string;
    type: CourseType;
};

// Визначаємо тип для заняття
type Lesson = {
    courseId: number;
    professorId: number;
    classroomNumber: string;
    dayOfWeek: DayOfWeek;
    timeSlot: TimeSlot;
};


// Крок 3: Робота з масивами даних


// Масиви даних
let professors: Professor[] = [];
let classrooms: Classroom[] = [];
let courses: Course[] = [];
let schedule: Lesson[] = [];

// Функція для додавання професора
function addProfessor(professor: Professor): void {
    professors.push(professor);
}

// Функція для додавання заняття
function addLesson(lesson: Lesson): boolean {
    // Перевіряємо на конфлікт часу з існуючим розкладом
    const hasConflict = schedule.some(existingLesson =>
        existingLesson.dayOfWeek === lesson.dayOfWeek &&
        existingLesson.timeSlot === lesson.timeSlot &&
        (existingLesson.classroomNumber === lesson.classroomNumber || 
         existingLesson.professorId === lesson.professorId)
    );

    if (!hasConflict) {
        schedule.push(lesson);
        return true;  // Заняття успішно додано
    }
    return false;  // Конфлікт, заняття не додано
}


// Крок 4: Функції пошуку та фільтрації


// Функція для пошуку вільних аудиторій
function findAvailableClassrooms(timeSlot: TimeSlot, dayOfWeek: DayOfWeek): string[] {
    // Знаходимо всі зайняті аудиторії в цей час
    const occupiedClassrooms = schedule
        .filter(lesson => lesson.timeSlot === timeSlot && lesson.dayOfWeek === dayOfWeek)
        .map(lesson => lesson.classroomNumber);
    
    // Повертаємо аудиторії, які не зайняті
    return classrooms
        .filter(classroom => !occupiedClassrooms.includes(classroom.number))
        .map(classroom => classroom.number);
}

// Функція для отримання розкладу професора
function getProfessorSchedule(professorId: number) {
    const professorLessons = schedule.filter(function (lesson) {
        return lesson.professorId === professorId;
    });

    if (professorLessons.length === 0) {
        console.log(`Професор з id ${professorId} не має запланованих занять`);
    } else {
        return professorLessons;
    }
}


// Крок 5: Обробка конфліктів та валідація


// Визначаємо тип конфлікту в розкладі
type ScheduleConflict = {
    type: "ProfessorConflict" | "ClassroomConflict";
    lessonDetails: Lesson;
};

// Функція для валідації заняття
function validateLesson(lesson: Lesson): ScheduleConflict | null {
    // Перевіряємо на конфлікт з іншим заняттям у цей час
    for (const existingLesson of schedule) {
        if (existingLesson.dayOfWeek === lesson.dayOfWeek && existingLesson.timeSlot === lesson.timeSlot) {
            if (existingLesson.classroomNumber === lesson.classroomNumber) {
                return {
                    type: "ClassroomConflict",
                    lessonDetails: existingLesson
                };
            }
            if (existingLesson.professorId === lesson.professorId) {
                return {
                    type: "ProfessorConflict",
                    lessonDetails: existingLesson
                };
            }
        }
    }
    return null; // Конфліктів немає
}


// Крок 6: Аналіз та звіти


// Функція для розрахунку відсотка використання аудиторії
function getClassroomUtilization(classroomNumber: string): number {
    const totalSlots = 5 * 5; // 5 днів на тиждень по 5 тайм-слотів кожного дня
    const occupiedSlots = schedule.filter(lesson => lesson.classroomNumber === classroomNumber).length;
    return (occupiedSlots / totalSlots) * 100;
}

// Функція для знаходження найпопулярнішого типу занять
function getMostPopularCourseType(): CourseType {
    const typeCount: Record<CourseType, number> = {
        Lecture: 0,
        Seminar: 0,
        Lab: 0,
        Practice: 0
    };

    // Підраховуємо кількість кожного типу заняття
    schedule.forEach(lesson => {
        const course = courses.find(course => course.id === lesson.courseId);
        if (course) {
            typeCount[course.type]++;
        }
    });

    // Знаходимо тип з найбільшою кількістю занять
    return Object.keys(typeCount).reduce((a, b) => 
        typeCount[a as CourseType] > typeCount[b as CourseType] ? a : b
    ) as CourseType;
}


// Крок 7: Модифікація даних


// Функція для зміни аудиторії заняття
function reassignClassroom(lessonId: number, newClassroomNumber: string): boolean {
    const lesson = schedule.find(lesson => lesson.courseId === lessonId);
    
    if (lesson) {
        // Перевіряємо, чи не зайнята нова аудиторія в цей же час
        const conflict = schedule.some(existingLesson => 
            existingLesson.dayOfWeek === lesson.dayOfWeek &&
            existingLesson.timeSlot === lesson.timeSlot &&
            existingLesson.classroomNumber === newClassroomNumber
        );

        if (!conflict) {
            lesson.classroomNumber = newClassroomNumber;
            return true;  // Зміна успішна
        }
    }
    return false;  // Конфлікт або заняття не знайдено
}

// Функція для скасування заняття
function cancelLesson(lessonId: number): void {
    const lessonIndex = schedule.findIndex(lesson => lesson.courseId === lessonId);
    
    if (lessonIndex !== -1) {
        schedule.splice(lessonIndex, 1);  // Видаляємо заняття з розкладу
    }
}






// Приклад використання: Крок 4 Функції пошуку та фільтрації


// Додаємо деяких професорів
addProfessor({ id: 1, name: "John Doe", department: "Computer Science" });
addProfessor({ id: 2, name: "Jane Smith", department: "Mathematics" });

// Шукаємо доступні аудиторії на 8:30-10:00 у понеділок
const availableClassrooms = findAvailableClassrooms("8:30-10:00", "Monday");
console.log(availableClassrooms);  // Виведе номери доступних аудиторій

// Отримуємо розклад професора John Doe
const johnDoeSchedule = getProfessorSchedule(1);
console.log(johnDoeSchedule);  // Виведе масив занять професора



// Приклад використання: Крок 5 Обробка конфліктів та валідація


// Створюємо нове заняття
const newLesson: Lesson = {
    courseId: 1,
    professorId: 1,
    classroomNumber: "101",
    dayOfWeek: "Monday",
    timeSlot: "8:30-10:00"
};

// Перевіряємо заняття на конфлікти
const conflict = validateLesson(newLesson);

if (conflict) {
    console.log(`Конфлікт: ${conflict.type} під час ${conflict.lessonDetails.timeSlot}`);
} else {
    console.log("Заняття можна додати");
}


// Приклад використання: Крок 6: Аналіз та звіти


// Розрахунок використання аудиторії 101
const utilization = getClassroomUtilization("101");
console.log(`Використання аудиторії 101: ${utilization}%`);

// Найпопулярніший тип заняття
const popularCourseType = getMostPopularCourseType();
console.log(`Найпопулярніший тип заняття: ${popularCourseType}`);


// Приклад використання: Крок 7: Модифікація даних


// Спроба змінити аудиторію для заняття з id 1 на нову аудиторію 102
const reassigned = reassignClassroom(1, "102");
console.log(reassigned ? "Аудиторію змінено успішно" : "Не вдалося змінити аудиторію");

// Скасування заняття з id 1
cancelLesson(1);
console.log("Заняття скасовано");
