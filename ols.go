package ols

import (
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

// Model handles the regression
type Model struct {
	x            *mat64.Dense
	y            *mat64.Vector // one dimentional
	hasIntercept bool
	cols         int
	rows         int
}

// NewModel creates a new model with an intercept by default
func NewModel(x, y mat64.Matrix) *Model {
	return NewModelWithIntercept(x, y, true)
}

// NewModelWithIntercept creates a new OLS model with or without an intercept
// x or y can be nil, just needs to be updated before training
func NewModelWithIntercept(x, y mat64.Matrix, intercept bool) *Model {
	m := new(Model)
	m.hasIntercept = intercept
	m.SetX(x)
	m.SetY(y)

	return m
}

// SetX replaces the independent variables in the model
func (m *Model) SetX(data mat64.Matrix) {
	if data == nil {
		return
	}
	rows, cols := data.Dims()
	offset := 0
	if m.hasIntercept {
		offset = 1
	}
	m.cols = cols + offset
	m.rows = rows
	m.x = mat64.NewDense(rows, cols+offset, nil)
	for c := 0; c < cols+offset; c++ {
		if c == 0 && m.hasIntercept {
			intercept := make([]float64, rows)
			floats.AddConst(1.0, intercept)
			m.x.SetCol(0, intercept)
		} else {
			m.x.SetCol(c, column(data, c-offset))
		}
	}
}

// SetY replaces the dependent model
// NOTE: only uses first column of the matrix
func (m *Model) SetY(y mat64.Matrix) {
	if y == nil {
		return
	}
	rows, _ := y.Dims()
	m.y = mat64.NewVector(rows, nil)
	for r := 0; r < rows; r++ {
		m.y.SetVec(r, y.At(r, 0))
	}
}

// Dims returns number of independent variabes and count of rows
func (m *Model) Dims() (int, int) {
	return m.x.Dims()
}

// Train - process the data and return the result
func (m *Model) Train() *mat64.Dense {
	xt := m.x.T()
	sq := mat64.NewDense(m.cols, m.cols, nil)
	sq.Mul(xt, m.x)
	sq.Inverse(sq)
	res := mat64.NewDense(m.cols, 1, nil)
	res.Product(sq, xt, m.y)
	return res
}

func column(m mat64.Matrix, c int) []float64 {
	rows, _ := m.Dims()
	col := make([]float64, rows)
	mat64.Col(col, c, m)
	return col
}

func sum(data []float64) float64 {
	return floats.Sum(data)
}
