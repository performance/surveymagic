$(document).ready(function(){
    // Fetch and display the report
    $.getJSON("/api/report", function(data) {
        let reportHtml = "";
        reportHtml += `<h2>${data.report_title}</h2>`;
        reportHtml += `<h3>Executive Summary</h3>`;
        reportHtml += `<p>${data.executive_summary}</p>`;

        reportHtml += "<h3>Insights by Objective</h3>";
        data.insights_by_objective.forEach(function(insight) {
            reportHtml += `<button class=\"collapsible\">${insight.objectiveText}</button>`;
            reportHtml += `<div class=\"content\">`;
            reportHtml += `<p>${insight.synthesis}</p>`;
            reportHtml += "<h4>Supporting Analyses</h4>";
            insight.supportingAnalyses.forEach(function(analysis) {
                reportHtml += `<p><b>${analysis.question_text}</b>: ${analysis.headline}</p>`;
            });
            reportHtml += `</div>`;
        });

        reportHtml += "<h3>Question Analyses</h3>";
        data.question_analyses.forEach(function(analysis) {
            reportHtml += `<button class=\"collapsible\">${analysis.question_text}</button>`;
            reportHtml += `<div class=\"content\">`;
            reportHtml += `<p><b>Headline:</b> ${analysis.headline}</p>`;
            reportHtml += `<p><b>Summary:</b> ${analysis.summary}</p>`;
            reportHtml += "<h4>Themes</h4>";
            analysis.themes.forEach(function(theme) {
                reportHtml += `<button class=\"collapsible\">${theme.theme_title} (${theme.participant_count} participants, ${theme.participant_percentage})</button>`;
                reportHtml += `<div class=\"content\">`;
                reportHtml += `<p>${theme.theme_description}</p>`;
                reportHtml += "<h5>Supporting Quotes</h5>";
                theme.supporting_quotes.forEach(function(quote) {
                    reportHtml += `<p><em>\"${quote.quote}\"</em> - Participant ${quote.participantId}</p>`;
                });
                reportHtml += `</div>`;
            });
            reportHtml += `</div>`;
        });

        $("#report-container").html(reportHtml);

        // Make the collapsible sections work
        var coll = document.getElementsByClassName("collapsible");
        var i;

        for (i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
    });

    // Fetch and display the log file
    $.get("/api/log", function(data) {
        $("#log-content").text(data);
    });
});
