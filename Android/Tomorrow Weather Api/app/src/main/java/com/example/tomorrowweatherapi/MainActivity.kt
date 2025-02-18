package com.example.tomorrowweatherapi

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.tomorrowweatherapi.ui.theme.TomorrowWeatherApiTheme
import okhttp3.OkHttpClient
import okhttp3.Request
import kotlinx.coroutines.*
import java.io.IOException

// Main activity class
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Sets the UI content for activity, defining the app's layout
        setContent {
            WeatherApp() // Calls the main WeatherApp composable function to build the UI
        }
    }
}

// A composable function to display the weather data
@Composable
fun WeatherApp() {
    var weatherData by remember { mutableStateOf("Loading...") }

    // LaunchedEffect to fetch weather data
    LaunchedEffect(Unit) {
        weatherData = fetchWeatherData() // Fetch weather data asynchronously
    }

    // UI layout, displaying the weather data or loading message
    Surface(color = MaterialTheme.colorScheme.background) {
        Text(text = weatherData, modifier = Modifier.fillMaxSize())
    }
}

// Suspend function to fetch weather data from the API asynchronously
suspend fun fetchWeatherData(): String = withContext(Dispatchers.IO) { // Perform network operation on IO dispatcher
    val client = OkHttpClient()

    // Build the request with the API URL, HTTP method, and headers
    val request = Request.Builder()
        .url("https://api.tomorrow.io/v4/weather/realtime?location=miami&apikey=Pc18198iI8Gyr7j9F5lPLioQleYyCenx")
        .get()
        .addHeader("accept", "application/json")
        .build()

    // Try to execute the request and handle the response or errors
    try {
        // Execute the request
        val response = client.newCall(request).execute()
        if (response.isSuccessful) { // Return the response body as a string if successful
            response.body?.string() ?: "Error: Empty response"
        } else {
            "Error: ${response.message}" // Return an error message if response is unsuccessful
        }
    } catch (e: IOException) {
        "Error: ${e.message}"// Catch and return any IO exceptions that occur
    }
}
@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    TomorrowWeatherApiTheme {
        Greeting("Android")
    }
}