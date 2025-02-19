// Enum - перелічувані типи для фіксованих значень
enum StudentStatus {
    Active = "Active", // Активний студент
    Academic_Leave = "Academic_Leave", // Академічна відпустка
    Graduated = "Graduated", // Випускник
    Expelled = "Expelled" // Відрахований
}

enum CourseType {
    Mandatory = "Mandatory", // Обов'язковий курс
    Optional = "Optional", // Вибірковий курс
    Special = "Special" // Спеціальний курс
}

enum Semester {
    First = "First", // Перший семестр
    Second = "Second" // Другий семестр
}

enum GradeValue {
    Excellent = 5, // Відмінно
    Good = 4, // Добре
    Satisfactory = 3, // Задовільно
    Unsatisfactory = 2 // Незадовільно
}

enum Faculty {
    Computer_Science = "Computer_Science", // Факультет комп'ютерних наук
    Economics = "Economics", // Економічний факультет
    Law = "Law", // Юридичний факультет
    Engineering = "Engineering" // Інженерний факультет
}

// Інтерфейс для опису структури студента
interface Student {
    id: number; // Унікальний ідентифікатор студента
    fullName: string; // Повне ім'я
    faculty: Faculty; // Факультет
    year: number; // Рік навчання
    status: StudentStatus; // Статус студента
    enrollmentDate: Date; // Дата зарахування
    groupNumber: string; // Номер групи
}

// Інтерфейс для опису структури курсу
interface Course {
    id: number; // Унікальний ідентифікатор курсу
    name: string; // Назва курсу
    type: CourseType; // Тип курсу
    credits: number; // Кількість кредитів
    semester: Semester; // Семестр
    faculty: Faculty; // Факультет
    maxStudents: number; // Максимальна кількість студентів
}

// Інтерфейс для опису оцінки студента
interface Grade {
    studentId: number; // ID студента
    courseId: number; // ID курсу
    grade: GradeValue; // Оцінка
    date: Date; // Дата виставлення оцінки
    semester: Semester; // Семестр
}

// Клас для управління університетом
class UniversityManagementSystem {
    private students: Student[] = []; // Масив студентів
    private courses: Course[] = []; // Масив курсів
    private grades: Grade[] = []; // Масив оцінок
    private nextStudentId = 1; // Наступний ID для студента
    private nextCourseId = 1; // Наступний ID для курсу

    // Метод для реєстрації студента
    enrollStudent(student: Omit<Student, "id">): Student {
        const newStudent: Student = { id: this.nextStudentId++, ...student };
        this.students.push(newStudent); // Додаємо студента до списку
        return newStudent; // Повертаємо зареєстрованого студента
    }

    // Метод для реєстрації студента на курс
    registerForCourse(studentId: number, courseId: number): void {
        const student = this.students.find(s => s.id === studentId);
        const course = this.courses.find(c => c.id === courseId);

        // Перевірки: чи існують студент і курс
        if (!student || !course) throw new Error("Student or course not found.");

        // Перевірка відповідності факультету студента і курсу
        if (student.faculty !== course.faculty) throw new Error("Student cannot register for a course outside their faculty.");

        // Перевірка, чи є вільні місця на курсі
        const studentCount = this.grades.filter(g => g.courseId === courseId).length;
        if (studentCount >= course.maxStudents) throw new Error("Course is full.");

        // Реєстрація студента на курс
        this.grades.push({ studentId, courseId, grade: GradeValue.Unsatisfactory, date: new Date(), semester: course.semester });
    }

    // Метод для виставлення оцінки студенту
    setGrade(studentId: number, courseId: number, grade: GradeValue): void {
        const gradeRecord = this.grades.find(g => g.studentId === studentId && g.courseId === courseId);

        // Перевірка, чи зареєстрований студент на курс
        if (!gradeRecord) throw new Error("Student is not registered for this course.");

        gradeRecord.grade = grade; // Встановлюємо оцінку
    }

    // Метод для оновлення статусу студента
    updateStudentStatus(studentId: number, newStatus: StudentStatus): void {
        const student = this.students.find(s => s.id === studentId);

        // Перевірка, чи існує студент
        if (!student) throw new Error("Student not found.");

        student.status = newStatus; // Оновлення статусу
    }

    // Отримання студентів за факультетом
    getStudentsByFaculty(faculty: Faculty): Student[] {
        return this.students.filter(s => s.faculty === faculty);
    }

    // Отримання оцінок студента
    getStudentGrades(studentId: number): Grade[] {
        return this.grades.filter(g => g.studentId === studentId);
    }

    // Отримання доступних курсів за факультетом і семестром
    getAvailableCourses(faculty: Faculty, semester: Semester): Course[] {
        return this.courses.filter(c => c.faculty === faculty && c.semester === semester);
    }

    // Обчислення середнього балу студента
    calculateAverageGrade(studentId: number): number {
        const studentGrades = this.getStudentGrades(studentId);

        // Якщо оцінок немає, повертаємо 0
        if (studentGrades.length === 0) return 0;

        // Сума всіх оцінок і обчислення середнього
        const total = studentGrades.reduce((sum, g) => sum + g.grade, 0);
        return total / studentGrades.length;
    }

    // Отримання списку відмінників по факультету
    getTopStudentsByFaculty(faculty: Faculty): Student[] {
        const students = this.getStudentsByFaculty(faculty);

        // Відбір студентів з середнім балом не нижче "Добре"
        return students.filter(student => this.calculateAverageGrade(student.id) >= GradeValue.Good);
    }
}
