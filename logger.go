package ols

import (
	"github.com/Sirupsen/logrus"
)

var log *logrus.Logger

func init() {
	SetLogger(logrus.New())
}

// SetLogger replaces the default logger
func SetLogger(logger *logrus.Logger) {
	log = logger
}
