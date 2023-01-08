rm(list = ls())
cat("\014")

# Directory
filepath <- rstudioapi::getSourceEditorContext()$path
dirpath <- dirname(rstudioapi::getSourceEditorContext()$path)

setwd(dirpath)

# Install packages and add to library
packages <- c("dplyr", "fst", "feather")

if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))
  }

invisible(lapply(packages, library, character.only  = TRUE))


# Load data
global_daily <- read.csv("crsp_us_daily.csv")

global_daily_trim <- global_daily %>%
  select(-SHRCD, -EXCHCD, -COMNAM, -CUSIP) %>%
  na.omit()

# By ticker filter to max date range
ticker_summary <- global_daily_trim %>%
  group_by(TICKER) %>%
  summarize(mind = min(date),
            maxd = max(date),
            lend = length(unique(date))) %>%
  filter(mind == 20000103,
         maxd == 20211231)

# Find out which number of dates is most frequent
ticker_n <- ticker_summary %>%
  group_by(lend) %>%
  tally()

# Make list of tickers with this frequency
tickers <- ticker_summary %>%
  filter(lend == ticker_n$lend[which.max(ticker_n$n)]) %>%
  select(TICKER)

# Filter data by ticker list
global_daily_filtered <- global_daily_trim %>%
  filter(TICKER %in% tickers$TICKER) %>%
  arrange(date, PERMNO, TICKER)

# Check that every permno,ticker has same dates, and use only these dates
date_summary <- global_daily_filtered %>%
  group_by(date) %>%
  summarize(dates = length(date),
            permnos = length(PERMNO),
            tickers = length(TICKER),
            dateslu = length(unique(date)),
            permnoslu = length(unique(PERMNO)),
            tickerslu = length(unique(TICKER)))

# Check again
ticker_summary2 <- global_daily_filtered %>%
  group_by(TICKER) %>%
  summarize(mind = min(date),
            maxd = max(date),
            lend = length(unique(date)))

ticker_n2 <- ticker_summary2 %>%
  group_by(lend) %>%
  tally()
# ok

daily_us <- global_daily_filtered %>%
  rename(permno = PERMNO,
         ticker = TICKER,
         low = BIDLO,
         high = ASKHI,
         close = PRC,
         volume = VOL,
         ret = RET,
         bid = BID,
         ask = ASK,
         open = OPENPRC,
         numtrd = NUMTRD,
         spret = sprtrn) %>%
  mutate(ret = as.numeric(ret))

summary(daily_us)

# Download as feather
write_feather(daily_us, "daily_us.feather")
