$(() => {
  $("#check").click(() => {
    fetch(`http://127.0.0.1:8000/${$("#content").val()}`)
      .then((res) => res.json())
      .then((result) => {
        if (result === "ham") {
          $(".result").text("You're Email is Safe.");
        } else {
          $(".result").text("You're Email is NOT Safe.");
        }
      });
  });
});
