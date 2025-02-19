
// Крок 1. Базова структура для різних типів контенту

// Розширюємо базовий інтерфейс BaseContent
interface BaseContent {
    id: string; // Унікальний ідентифікатор контенту
    createdAt: Date; // Дата створення
    updatedAt: Date; // Дата останнього оновлення
    publishedAt?: Date; // Опціональна дата публікації
    status: 'draft' | 'published' | 'archived'; // Статус контенту
  }
  
  // Інтерфейс для статей
  interface Article extends BaseContent {
    title: string; // Заголовок статті
    content: string; // Основний текст
    author: string; // Автор статті
  }
  
  // Інтерфейс для продуктів
  interface Product extends BaseContent {
    name: string; // Назва продукту
    price: number; // Ціна продукту
    description: string; // Опис
    stock: number; // Кількість на складі
  }

//   Створюємо тип для операцій із контентом

type ContentOperations<T extends BaseContent> = {
    create: (data: T) => T; // Створення нового контенту
    read: (id: string) => T | null; // Читання контенту за ID
    update: (id: string, data: Partial<T>) => T; // Оновлення контенту
    delete: (id: string) => boolean; // Видалення контенту
  };

//   Крок 2. Система управління правами доступу

// Визначаємо ролі та права
type Role = 'admin' | 'editor' | 'viewer';

type Permission = {
  create: boolean;
  read: boolean;
  update: boolean;
  delete: boolean;
};

// Система контролю доступу для різних типів контенту
type AccessControl<T extends BaseContent> = {
    role: Role; // Роль користувача
    permissions: Permission; // Права доступу
    canAccess: (operation: keyof Permission, content: T) => boolean; // Перевірка доступу
  };

//   Крок 3. Система валідації

// Базовий валідатор
type Validator<T> = {
    validate: (data: T) => ValidationResult;
  };
  
  type ValidationResult = {
    isValid: boolean;
    errors?: string[]; // Список помилок, якщо є
  };
  

//   Приклад валідатора для статей
const articleValidator: Validator<Article> = {
    validate: (data) => {
      const errors: string[] = [];
      if (!data.title || data.title.length < 5) {
        errors.push('Заголовок має містити мінімум 5 символів');
      }
      if (!data.content) {
        errors.push('Контент не може бути порожнім');
      }
      return { isValid: errors.length === 0, errors };
    },
  };

//   Крок 4. Система версіонування

// Тип для версіонованого контенту
type Versioned<T extends BaseContent> = T & {
    version: number; // Поточна версія
    changes: Array<{ // Історія змін
      version: number;
      updatedAt: Date;
      changes: Partial<T>;
    }>;
  };

//   Приклад реалізації версіонування для статті
const versionedArticle: Versioned<Article> = {
    id: '1',
    createdAt: new Date(),
    updatedAt: new Date(),
    publishedAt: new Date(),
    status: 'draft',
    title: 'Моя перша стаття',
    content: 'Це текст статті',
    author: 'Артем',
    version: 1,
    changes: [],
  };
  