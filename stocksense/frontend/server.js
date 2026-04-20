const express = require("express");

const app = express();
const PORT = process.env.PORT || 5500;

app.set("view engine", "ejs");
app.set("views", `${__dirname}/views`);
app.use(express.static(`${__dirname}/public`));

app.get("/", (_req, res) => {
  res.render("index", {
    appTitle: "StockSense Dashboard",
  });
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.listen(PORT, () => {
  console.log(`StockSense frontend running on http://127.0.0.1:${PORT}`);
});
