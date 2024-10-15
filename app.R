# Load necessary libraries
library(shiny)
library(shinydashboard)
library(tidyverse)
library(lubridate)
library(plotly)
library(DT)
library(forecast)
library(bcrypt)
library(httr)
library(jsonlite)
library(keras)
library(shinyjs)
library(shinyauthr)
library(leaflet)
library(rsconnect)



# Helper Functions

# Data Acquisition with Error Handling
# This function fetches COVID-19 data from a remote CSV file and processes it into a useful format.
fetch_covid_data <- function() {
  tryCatch({
    url <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    data <- read_csv(url)
    
    # Transform the data to long format and convert dates
    data_long <- data %>%
      pivot_longer(cols = -c(`Province/State`, `Country/Region`, Lat, Long),
                   names_to = "date",
                   values_to = "confirmed") %>%
      mutate(date = as.Date(date, format = "%m/%d/%y"))
    
    # Group data by Country/Region and date, summing confirmed cases
    data_grouped <- data_long %>%
      group_by(`Country/Region`, date, Lat, Long) %>%
      summarise(confirmed = sum(confirmed), .groups = "drop")
    
    return(data_grouped)
  }, error = function(e) {
    message("Error fetching COVID-19 data: ", e$message)
    return(NULL)
  })
}

# LSTM Forecasting Model
# This function trains an LSTM model on confirmed COVID-19 cases for a specific country and makes predictions.
train_lstm_model <- function(data, country, days_to_predict = 30) {
  # Filter data for the specified country and arrange by date
  country_data <- data %>%
    filter(`Country/Region` == country) %>%
    arrange(date)
  
  X <- country_data$confirmed
  X <- scale(X)  # Standardize the data
  
  # Create sequences of data for LSTM input
  n_timesteps <- 7
  n_features <- 1
  X_seq <- array(0, dim = c(length(X) - n_timesteps, n_timesteps, n_features))
  y_seq <- array(0, dim = c(length(X) - n_timesteps))
  
  for (i in 1:(length(X) - n_timesteps)) {
    X_seq[i,,] <- X[i:(i+n_timesteps-1)]
    y_seq[i] <- X[i+n_timesteps]
  }
  
  # Build and train LSTM model
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(n_timesteps, n_features)) %>%
    layer_dense(units = 1)
  
  model %>% compile(optimizer = optimizer_adam(), loss = "mse")
  model %>% fit(X_seq, y_seq, epochs = 100, batch_size = 32, verbose = 0)
  
  # Make predictions
  last_sequence <- tail(X, n_timesteps)
  predictions <- numeric(days_to_predict)
  
  for (i in 1:days_to_predict) {
    next_pred <- model %>% predict(array(last_sequence, dim = c(1, n_timesteps, 1)))
    predictions[i] <- next_pred
    last_sequence <- c(last_sequence[-1], next_pred)
  }
  
  # Reverse the standardization
  predictions <- predictions * attr(X, "scaled:scale") + attr(X, "scaled:center")
  return(predictions)
}

# UI setup
# Define the structure and layout of the Shiny dashboard
ui <- dashboardPage(
  dashboardHeader(title = "Advanced COVID-19 Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Global Overview", tabName = "overview", icon = icon("globe")),
      menuItem("Country Analysis", tabName = "country_analysis", icon = icon("chart-line")),
      menuItem("Forecasting", tabName = "forecast", icon = icon("chart-area")),
      menuItem("Scenario Analysis", tabName = "scenario_analysis", icon = icon("project-diagram")),
      menuItem("Data Explorer", tabName = "data_explorer", icon = icon("table"))
    )
  ),
  dashboardBody(
    tabItems(
      # Global Overview tab
      tabItem(tabName = "overview",
              fluidRow(
                box(leafletOutput("global_map"), width = 12)  # Output the global map
              )
      ),
      # Country Analysis tab
      tabItem(tabName = "country_analysis",
              fluidRow(
                box(selectInput("country", "Select Country:", choices = NULL), width = 4),
                box(plotlyOutput("country_trend"), width = 8)
              )
      ),
      # Forecasting tab
      tabItem(tabName = "forecast",
              fluidRow(
                box(selectInput("forecast_country", "Select Country:", choices = NULL), width = 4),
                box(numericInput("forecast_days", "Days to Forecast:", value = 30, min = 1), width = 4),
                box(plotlyOutput("forecast_plot"), width = 12)
              )
      ),
      # Scenario Analysis tab
      tabItem(tabName = "scenario_analysis",
              fluidRow(
                box(selectInput("scenario_country", "Select Country:", choices = NULL), width = 4),
                box(numericInput("intervention_effect", "Intervention Effect (%):", value = 10, min = -100, max = 100), width = 4),
                box(dateInput("intervention_date", "Intervention Date:"), width = 4),
                box(plotlyOutput("scenario_plot"), width = 12)
              )
      ),
      # Data Explorer tab
      tabItem(tabName = "data_explorer",
              fluidRow(
                box(DTOutput("data_table"), width = 12)
              )
      )
    )
  )
)

# Server logic
# Define the server side logic for the Shiny app
server <- function(input, output, session) {
  covid_data <- reactiveVal()
  
  # Observe and update data and input choices
  observe({
    data <- fetch_covid_data()
    if (!is.null(data)) {
      covid_data(data)
      updateSelectInput(session, "country", choices = unique(data$`Country/Region`))
      updateSelectInput(session, "forecast_country", choices = unique(data$`Country/Region`))
      updateSelectInput(session, "scenario_country", choices = unique(data$`Country/Region`))
    }
  })
  
  # Render the global map
  output$global_map <- renderLeaflet({
    req(covid_data())
    data <- covid_data()
    leaflet(data) %>%
      addTiles() %>%
      addCircles(lng = ~Long, lat = ~Lat, weight = 1,
                 radius = ~sqrt(confirmed) * 500, popup = ~`Country/Region`)
  })
  
  # Render the country trend plot
  output$country_trend <- renderPlotly({
    req(input$country, covid_data())
    data <- covid_data() %>% filter(`Country/Region` == input$country)
    plot_ly(data, x = ~date, y = ~confirmed, type = "scatter", mode = "lines") %>%
      layout(title = paste("COVID-19 Trend for", input$country),
             xaxis = list(title = "Date"),
             yaxis = list(title = "Total Confirmed Cases"))
  })
  
  # Render the forecast plot
  output$forecast_plot <- renderPlotly({
    req(input$forecast_country, covid_data())
    
    country_data <- covid_data() %>% 
      filter(`Country/Region` == input$forecast_country) %>%
      arrange(date)
    
    forecast_result <- train_lstm_model(country_data, input$forecast_country, input$forecast_days)
    
    last_date <- max(country_data$date)
    forecast_dates <- seq(last_date + 1, by = "day", length.out = input$forecast_days)
    
    plot_ly() %>%
      add_trace(x = country_data$date, y = country_data$confirmed, 
                name = "Actual", type = "scatter", mode = "lines") %>%
      add_trace(x = forecast_dates, y = forecast_result, 
                name = "Forecast", type = "scatter", mode = "lines") %>%
      layout(title = paste("COVID-19 Forecast for", input$forecast_country),
             xaxis = list(title = "Date"),
             yaxis = list(title = "Total Confirmed Cases"))
  })
  
  # Render the scenario analysis plot
  output$scenario_plot <- renderPlotly({
    req(input$scenario_country, input$intervention_effect, input$intervention_date, covid_data())
    
    country_data <- covid_data() %>%
      filter(`Country/Region` == input$scenario_country) %>%
      arrange(date)
    
    # Apply the intervention effect on cases starting from the intervention date
    intervention_date <- as.Date(input$intervention_date)
    effect <- 1 + (input$intervention_effect / 100)
    
    country_data <- country_data %>%
      mutate(
        ScenarioConfirmed = if_else(date >= intervention_date, confirmed * effect, confirmed)
      )
    
    plot_ly() %>%
      add_trace(x = country_data$date, y = country_data$confirmed, 
                name = "Actual", type = "scatter", mode = "lines") %>%
      add_trace(x = country_data$date, y = country_data$ScenarioConfirmed, 
                name = "Scenario", type = "scatter", mode = "lines") %>%
      layout(title = paste("COVID-19 Scenario Analysis for", input$scenario_country),
             xaxis = list(title = "Date"),
             yaxis = list(title = "Total Confirmed Cases"))
  })
  
  # Render the data table
  output$data_table <- renderDT({
    req(covid_data())
    datatable(covid_data(), options = list(pageLength = 10, scrollX = TRUE))  # Updated with scrollX and pageLength
  })
}

# Run the application 
shinyApp(ui = ui, server = server)

